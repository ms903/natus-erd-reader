"""Optional, dependency-free CPython acceleration (no NumPy C ABI)."""
import os
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class NativeBuild(build_ext):
    def build_extensions(self):
        for extension in self.extensions:
            extension.extra_compile_args = (["/O2", "/fp:strict", "/std:c11"]
                if self.compiler.compiler_type == "msvc" else
                ["-O3", "-fno-fast-math", "-ffp-contract=off"])
        super().build_extensions()

extensions = []
if os.environ.get("NATUS_ERD_NO_NATIVE") != "1":
    extensions.append(Extension(
        "natus_erd._native", ["src/natus_erd/_native.c"],
        optional=os.environ.get("NATUS_ERD_REQUIRE_NATIVE") != "1",
    ))
setup(ext_modules=extensions, cmdclass={"build_ext": NativeBuild})
