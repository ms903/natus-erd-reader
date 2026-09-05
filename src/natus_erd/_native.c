#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <string.h>

/* Only immutable bytes/tuples and checked writable contiguous buffers cross
 * this boundary. All Python objects are inspected while holding the GIL.
 * No Python API, allocation, or shared mutable state in the released section.
 * Schema 9 packets restart with absolute values; states never cross packets. */
typedef struct {
    int raw, dmin, dmax;
    double source_scale, pmin, pmax;
    int64_t raw_min, raw_step;
} Calibration;

static double even_round(double x) {
    double f = floor(x), remainder = x - f;
    if (remainder > 0.5 || (remainder == 0.5 && fmod(f, 2.0) != 0.0)) f += 1.0;
    return f;
}
static int run(const uint8_t *p, size_t length, int count, int n,
               const uint8_t *shorted, int start, int stop, const int *selected,
               int rows, int operation, const Calibration *cal, uint8_t *out,
               size_t width, size_t column, int64_t *low, int64_t *high,
               double *errors) {
    int64_t state[1024] = {0};
    int absolutes[1024];
    size_t pos = 0, mask_size = (size_t)(n + 7) / 8;
    for (int sample = 0; sample < stop; ++sample) {
        if (length - pos < 1 + mask_size) return 1;
        if (p[pos++] > 1) return 2;
        const uint8_t *mask = p + pos; pos += mask_size;
        int absolute_count = 0, active = 0;
        for (int c = 0; c < n; ++c) {
            if (shorted[c]) continue;
            ++active;
            int64_t delta;
            if (mask[c >> 3] & (1u << (c & 7))) {
                if (length - pos < 2) return 1;
                uint32_t u = (uint32_t)p[pos] | ((uint32_t)p[pos + 1] << 8);
                pos += 2;
                delta = u >= 32768 ? (int64_t)u - 65536 : (int64_t)u;
                if (delta == -1) { absolutes[absolute_count++] = c; continue; }
            } else {
                if (length - pos < 1) return 1;
                uint32_t u = p[pos++];
                delta = u >= 128 ? (int64_t)u - 256 : (int64_t)u;
            }
            if (sample == 0) return 3;
            if ((delta > 0 && state[c] > INT64_MAX - delta) ||
                (delta < 0 && state[c] < INT64_MIN - delta)) return 4;
            state[c] += delta;
        }
        for (int i = 0; i < absolute_count; ++i) {
            if (length - pos < 4) return 1;
            uint32_t u = (uint32_t)p[pos] | ((uint32_t)p[pos+1] << 8) |
                         ((uint32_t)p[pos+2] << 16) | ((uint32_t)p[pos+3] << 24);
            state[absolutes[i]] = u >= UINT32_C(2147483648) ?
                (int64_t)u - INT64_C(4294967296) : (int64_t)u;
            pos += 4;
        }
        if (sample == 0 && absolute_count != active) return 3;
        if (sample < start) continue;
        for (int row = 0; row < rows; ++row) {
            int c = selected[row];
            if (shorted[c]) {
                if (operation == 2) {
                    size_t at = ((size_t)row * width + column + (size_t)(sample-start)) * 2;
                    out[at] = 255; out[at+1] = 127;
                }
                continue;
            }
            int64_t value = state[c];
            if (operation == 0) {
                if (sample == start) low[row] = high[row] = value;
                if (value < low[row]) low[row] = value;
                if (value > high[row]) high[row] = value;
            } else if (operation == 1) {
                double v = (double)value;
                size_t at = ((size_t)row * width + column + (size_t)(sample-start)) * 8;
                memcpy(out + at, &v, 8);
            } else {
                const Calibration *k = &cal[row];
                int64_t code;
                if (k->raw) {
                    int64_t diff = value - k->raw_min;
                    if (diff < 0 || diff % k->raw_step) return 5;
                    code = diff / k->raw_step + k->dmin;
                } else {
                    double physical = (double)value * k->source_scale;
                    if (!isfinite(physical) || physical < fmin(k->pmin, k->pmax) || physical > fmax(k->pmin, k->pmax)) return 5;
                    double q = (physical-k->pmin) * ((double)(k->dmax-k->dmin)/(k->pmax-k->pmin)) + k->dmin;
                    double rounded = even_round(q);
                    if (!isfinite(rounded) || rounded < k->dmin || rounded > k->dmax) return 5;
                    code = (int64_t)rounded;
                    double reconstructed = ((double)code-k->dmin) *
                        ((k->pmax-k->pmin)/(double)(k->dmax-k->dmin)) + k->pmin;
                    double error = fabs(reconstructed-physical);
                    if (error > errors[row]) errors[row] = error;
                }
                if (code < k->dmin || code > k->dmax || code < -32768 || code > 32767) return 5;
                size_t at = ((size_t)row * width + column + (size_t)(sample-start)) * 2;
                uint16_t bits = (uint16_t)code;
                out[at] = (uint8_t)(bits & 255); out[at+1] = (uint8_t)(bits >> 8);
            }
        }
    }
    if (stop == count && pos != length) return 6;
    return 0;
}

static int exact_int(PyObject *object, long long *result) {
    if (!PyLong_CheckExact(object)) {
        PyErr_SetString(PyExc_ValueError, "Native integer arguments must be exact integers"); return 0;
    }
    *result = PyLong_AsLongLong(object);
    return !PyErr_Occurred();
}

static PyObject *process(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *payload, *shorts, *selection, *parameters, *output;
    int count, start, stop, operation;
    Py_ssize_t width, column;
    if (!PyArg_ParseTuple(args, "OOOiiiiOOnn", &payload, &shorts, &selection,
                          &count, &start, &stop, &operation, &parameters, &output,
                          &width, &column)) return NULL;
    if (!PyBytes_CheckExact(payload) || !PyBytes_CheckExact(shorts) || !PyTuple_CheckExact(selection)) {
        PyErr_SetString(PyExc_ValueError, "Native inputs must be immutable bytes and tuples"); return NULL;
    }
    Py_ssize_t n = PyBytes_GET_SIZE(shorts), rows = PyTuple_GET_SIZE(selection);
    Py_ssize_t length = PyBytes_GET_SIZE(payload);
    if (n < 1 || n > 1024 || rows < 1 || rows > 1024 || count < 1 || count > 32767 ||
        start < 0 || start >= stop || stop > count || operation < 0 || operation > 2 ||
        width < stop-start || column < 0 || column > width-(stop-start)) {
        PyErr_SetString(PyExc_ValueError, "Invalid native packet dimensions"); return NULL;
    }
    const uint8_t *shorted = (const uint8_t *)PyBytes_AS_STRING(shorts);
    int active = 0, selected[1024];
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (shorted[i] > 1) { PyErr_SetString(PyExc_ValueError, "Invalid shorted mask"); return NULL; }
        if (!shorted[i]) ++active;
    }
    Py_ssize_t per_sample = 1+(n+7)/8+6*active;
    if (length < 1+(n+7)/8 || length > (Py_ssize_t)count*per_sample) {
        PyErr_SetString(PyExc_ValueError, "Invalid native payload length"); return NULL;
    }
    for (Py_ssize_t row = 0; row < rows; ++row) {
        long long c;
        if (!exact_int(PyTuple_GET_ITEM(selection, row), &c)) return NULL;
        if (c < 0 || c >= n) { PyErr_SetString(PyExc_ValueError, "Invalid native channel index"); return NULL; }
        selected[row] = (int)c;
    }
    Calibration cal[1024]; memset(cal, 0, sizeof(cal));
    if (operation == 2) {
        if (!PyTuple_CheckExact(parameters) || PyTuple_GET_SIZE(parameters) != rows) {
            PyErr_SetString(PyExc_ValueError, "Invalid native calibration table"); return NULL;
        }
        for (Py_ssize_t row = 0; row < rows; ++row) {
            PyObject *v = PyTuple_GET_ITEM(parameters, row);
            if (!PyTuple_CheckExact(v) || PyTuple_GET_SIZE(v) != 8) {
                PyErr_SetString(PyExc_ValueError, "Invalid native calibration row"); return NULL;
            }
            long long raw, dmin, dmax, rmin, step;
            if (!exact_int(PyTuple_GET_ITEM(v,0), &raw) || !exact_int(PyTuple_GET_ITEM(v,4), &dmin) ||
                !exact_int(PyTuple_GET_ITEM(v,5), &dmax) || !exact_int(PyTuple_GET_ITEM(v,6), &rmin) ||
                !exact_int(PyTuple_GET_ITEM(v,7), &step)) return NULL;
            double scale = PyFloat_AsDouble(PyTuple_GET_ITEM(v,1));
            double pmin = PyFloat_AsDouble(PyTuple_GET_ITEM(v,2));
            double pmax = PyFloat_AsDouble(PyTuple_GET_ITEM(v,3));
            if (PyErr_Occurred()) return NULL;
            if ((raw != 0 && raw != 1) || !isfinite(scale) || !isfinite(pmin) || !isfinite(pmax) ||
                pmin == pmax || dmin < -32768 || dmax > 32767 || dmin >= dmax || step < 1 ||
                rmin < -INT64_C(9007199254740991) || rmin > INT64_C(9007199254740991) ||
                step > INT64_C(9007199254740991)) {
                PyErr_SetString(PyExc_ValueError, "Unsafe native calibration"); return NULL;
            }
            cal[row] = (Calibration){(int)raw,(int)dmin,(int)dmax,scale,pmin,pmax,rmin,step};
        }
    }
    Py_buffer buffer = {0};
    if (operation != 0) {
        size_t item = operation == 1 ? 8 : 2;
        if ((size_t)width > (size_t)PY_SSIZE_T_MAX/(size_t)rows/item) {
            PyErr_SetString(PyExc_ValueError, "Native output size overflow"); return NULL;
        }
        if (PyObject_GetBuffer(output, &buffer, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) return NULL;
        if (buffer.len != width*rows*(Py_ssize_t)item) {
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "Native output buffer size mismatch"); return NULL;
        }
    }
    int64_t low[1024] = {0}, high[1024] = {0};
    double errors[1024] = {0};
    const uint8_t *p = (const uint8_t *)PyBytes_AS_STRING(payload);
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = run(p,(size_t)length,count,(int)n,shorted,start,stop,selected,(int)rows,
                 operation,cal,(uint8_t *)buffer.buf,(size_t)width,(size_t)column,low,high,errors);
    Py_END_ALLOW_THREADS
    if (buffer.obj) PyBuffer_Release(&buffer);
    if (status) { PyErr_Format(PyExc_ValueError, "Native schema-9 integrity check failed (%d)", status); return NULL; }
    if (operation == 1) Py_RETURN_NONE;
    PyObject *result = PyTuple_New(rows);
    if (!result) return NULL;
    for (Py_ssize_t row = 0; row < rows; ++row) {
        PyObject *item = operation == 0 ? Py_BuildValue("LL", (long long)low[row], (long long)high[row]) : PyFloat_FromDouble(errors[row]);
        if (!item) { Py_DECREF(result); return NULL; }
        PyTuple_SET_ITEM(result,row,item);
    }
    return result;
}
static PyMethodDef methods[] = {
    {"process",process,METH_VARARGS,"Checked schema-9 reduction, decode, or EDF encoding."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,"_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
