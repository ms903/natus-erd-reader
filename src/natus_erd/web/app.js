"use strict";

const state = {
  info: null,
  data: null,
  selected: [],
  mode: "overlay",
  request: null,
};

const elements = {};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  [
    "connectionStatus", "startInput", "durationSelect", "channelButton",
    "previousButton", "nextButton", "loadButton", "channelChips", "timeline",
    "timelineStart", "timelineEnd", "windowSummary", "sampleSummary",
    "alignmentSummary", "labelSummary", "lsbSummary", "clippedSummary",
    "uvSummary", "workspace", "plotLoading", "overlayPlots", "splitPlots",
    "overlayCanvas", "erdCanvas", "edfCanvas", "diffCanvas", "overlayMode", "splitMode",
    "eventList", "channelDialog", "channelSearch", "channelOptions",
    "selectionHint", "applyChannels", "toast"
  ].forEach((id) => { elements[id] = document.getElementById(id); });

  bindControls();
  try {
    const response = await fetch("/api/info", { cache: "no-store" });
    if (!response.ok) throw new Error("无法读取记录信息");
    state.info = await response.json();
    state.selected = [...state.info.defaultChannels];
    applyInfo();
    renderChannelChips();
    renderChannelOptions();
    setConnection(true, `${state.info.sampleRate} Hz · ${state.info.segments} 个 ERD 分段`);
    await loadWindow();
  } catch (error) {
    setConnection(false, "本地记录连接失败");
    showError(error.message || String(error));
  }
}

function bindControls() {
  elements.loadButton.addEventListener("click", loadWindow);
  elements.previousButton.addEventListener("click", () => moveWindow(-1));
  elements.nextButton.addEventListener("click", () => moveWindow(1));
  elements.startInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadWindow();
  });
  elements.durationSelect.addEventListener("change", loadWindow);
  elements.timeline.addEventListener("input", () => {
    elements.startInput.value = Number(elements.timeline.value).toFixed(2);
  });
  elements.timeline.addEventListener("change", loadWindow);
  elements.channelButton.addEventListener("click", () => {
    elements.channelSearch.value = "";
    renderChannelOptions();
    elements.channelDialog.showModal();
    setTimeout(() => elements.channelSearch.focus(), 0);
  });
  elements.channelSearch.addEventListener("input", renderChannelOptions);
  elements.applyChannels.addEventListener("click", (event) => {
    event.preventDefault();
    if (!state.selected.length) {
      showError("请至少选择一个通道");
      return;
    }
    elements.channelDialog.close();
    renderChannelChips();
    loadWindow();
  });
  elements.overlayMode.addEventListener("click", () => setMode("overlay"));
  elements.splitMode.addEventListener("click", () => setMode("split"));
  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawAll, 120);
  }).observe(elements.workspace);
}

function applyInfo() {
  const duration = state.info.durationSeconds;
  elements.timeline.max = Math.max(0, duration - Number(elements.durationSelect.value));
  elements.timelineEnd.textContent = formatClock(duration);
  elements.alignmentSummary.textContent = `${state.info.sampleRate} Hz · ${state.info.nSamples.toLocaleString()} 点`;
  elements.labelSummary.textContent = `${state.info.labelMatches} / 256 个通道名称一致`;
  elements.channelButton.textContent = `已选 ${state.selected.length} 个通道`;
}

async function loadWindow() {
  if (!state.info || !state.selected.length) return;
  const duration = Number(elements.durationSelect.value);
  const maximumStart = Math.max(0, state.info.durationSeconds - duration);
  const start = clamp(Number(elements.startInput.value) || 0, 0, maximumStart);
  elements.startInput.value = start.toFixed(2);
  elements.timeline.max = maximumStart;
  elements.timeline.value = start;
  elements.plotLoading.hidden = false;
  elements.loadButton.disabled = true;

  if (state.request) state.request.abort();
  state.request = new AbortController();
  const width = Math.max(800, Math.floor(elements.workspace.clientWidth * 2));
  const params = new URLSearchParams({
    start: String(start),
    duration: String(duration),
    channels: state.selected.join(","),
    points: String(Math.min(4000, width)),
  });
  try {
    const response = await fetch(`/api/window?${params}`, {
      cache: "no-store",
      signal: state.request.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "窗口读取失败");
    state.data = payload;
    updateSummaries();
    renderEvents();
    drawAll();
  } catch (error) {
    if (error.name !== "AbortError") showError(error.message || String(error));
  } finally {
    elements.plotLoading.hidden = true;
    elements.loadButton.disabled = false;
  }
}

function moveWindow(direction) {
  const duration = Number(elements.durationSelect.value);
  const current = Number(elements.startInput.value) || 0;
  elements.startInput.value = Math.max(0, current + direction * duration).toFixed(2);
  loadWindow();
}

function renderChannelChips() {
  elements.channelChips.replaceChildren();
  state.selected.forEach((index) => {
    const channel = state.info.channels[index];
    const chip = document.createElement("span");
    chip.className = "channel-chip";
    chip.append(document.createTextNode(`${channel.name} · ${index}`));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.ariaLabel = `移除 ${channel.name}`;
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      if (state.selected.length === 1) {
        showError("请至少保留一个通道");
        return;
      }
      state.selected = state.selected.filter((value) => value !== index);
      renderChannelChips();
      renderChannelOptions();
      loadWindow();
    });
    chip.append(remove);
    elements.channelChips.append(chip);
  });
  elements.channelButton.textContent = `已选 ${state.selected.length} 个通道`;
}

function renderChannelOptions() {
  if (!state.info) return;
  const query = elements.channelSearch.value.trim().toLocaleLowerCase();
  const fragment = document.createDocumentFragment();
  const matching = state.info.channels.filter((channel) =>
    !query || channel.name.toLocaleLowerCase().includes(query) || String(channel.index).includes(query)
  );
  matching.forEach((channel) => {
    const label = document.createElement("label");
    label.className = `channel-option${channel.shorted ? " shorted" : ""}`;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.includes(channel.index);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        if (state.selected.length >= state.info.maxChannels) {
          checkbox.checked = false;
          showError(`最多同时显示 ${state.info.maxChannels} 个通道`);
          return;
        }
        state.selected.push(channel.index);
      } else {
        state.selected = state.selected.filter((value) => value !== channel.index);
      }
      elements.selectionHint.textContent = `已选择 ${state.selected.length} / ${state.info.maxChannels}`;
    });
    const text = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = channel.name;
    const small = document.createElement("small");
    small.textContent = `通道 ${channel.index}`;
    text.append(strong, small);
    label.append(checkbox, text);
    fragment.append(label);
  });
  elements.channelOptions.replaceChildren(fragment);
  elements.selectionHint.textContent = `已选择 ${state.selected.length} / ${state.info.maxChannels} · 共 ${matching.length} 个匹配`;
}

function setMode(mode) {
  state.mode = mode;
  const overlay = mode === "overlay";
  elements.overlayMode.setAttribute("aria-pressed", String(overlay));
  elements.splitMode.setAttribute("aria-pressed", String(!overlay));
  elements.overlayPlots.hidden = !overlay;
  elements.splitPlots.hidden = overlay;
  drawAll();
}

function updateSummaries() {
  const data = state.data;
  const end = data.startSeconds + data.durationSeconds;
  elements.windowSummary.textContent = `${formatClock(data.startSeconds)} – ${formatClock(end)}`;
  elements.sampleSummary.textContent = `${data.sourceSamples.toLocaleString()} 原始点 → ${data.displayPoints.toLocaleString()} 显示点`;
  elements.startInput.value = data.startSeconds.toFixed(2);
  elements.timeline.value = data.startSeconds;

  const metrics = data.channels.map((channel) => channel.metrics);
  const maxLsb = maximum(metrics.map((metric) => metric.maxLsb));
  const maxUv = maximum(metrics.map((metric) => metric.maxUv));
  const clipped = metrics.reduce((sum, metric) => sum + metric.clipped, 0);
  elements.lsbSummary.textContent = maxLsb == null ? "无有效样本" : `${formatNumber(maxLsb, 3)} LSB`;
  elements.clippedSummary.textContent = `${clipped.toLocaleString()} 点`;
  elements.uvSummary.textContent = maxUv == null ? "物理量误差 —" : `最大物理量误差 ${formatNumber(maxUv, 4)} µV`;
}

function renderEvents() {
  elements.eventList.replaceChildren();
  if (!state.data.events.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "当前窗口没有事件";
    elements.eventList.append(empty);
    return;
  }
  state.data.events.forEach((event) => {
    const item = document.createElement("div");
    item.className = "event-item";
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = `+${event.offsetSeconds.toFixed(3)}s`;
    const text = document.createElement("span");
    text.className = "event-text";
    text.textContent = event.text || "未命名事件";
    text.title = event.text || "未命名事件";
    item.append(time, text);
    elements.eventList.append(item);
  });
}

function drawAll() {
  if (!state.data) return;
  if (state.mode === "overlay") {
    drawPanel(elements.overlayCanvas, ["erd", "edf"]);
  } else {
    drawPanel(elements.erdCanvas, ["erd"]);
    drawPanel(elements.edfCanvas, ["edf"]);
  }
  drawPanel(elements.diffCanvas, ["diff"]);
}

function drawPanel(canvas, visibleSources) {
  const data = state.data;
  const isDifference = visibleSources.includes("diff");
  const rowHeight = isDifference ? 90 : 112;
  const top = 18;
  const bottom = 28;
  const cssWidth = Math.max(320, canvas.clientWidth || canvas.parentElement.clientWidth);
  const cssHeight = Math.max(isDifference ? 330 : 410, top + bottom + data.channels.length * rowHeight);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(cssWidth * ratio);
  canvas.height = Math.floor(cssHeight * ratio);
  canvas.style.height = `${cssHeight}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);

  const left = cssWidth < 560 ? 54 : 72;
  const right = 18;
  const plotWidth = cssWidth - left - right;
  const scaleSources = isDifference ? ["diff"] : ["erd", "edf"];
  context.font = '10px "Segoe UI", sans-serif';
  context.lineJoin = "round";
  context.lineCap = "round";

  data.channels.forEach((channel, row) => {
    const rowTop = top + row * rowHeight;
    const rowBottom = rowTop + rowHeight - 10;
    context.fillStyle = row % 2 ? "rgba(255,255,255,0.008)" : "rgba(255,255,255,0.018)";
    context.fillRect(left, rowTop, plotWidth, rowBottom - rowTop);

    context.strokeStyle = "rgba(49,80,100,0.26)";
    context.lineWidth = 1;
    for (let grid = 0; grid <= 8; grid += 1) {
      const x = left + (plotWidth * grid) / 8;
      context.beginPath();
      context.moveTo(x, rowTop);
      context.lineTo(x, rowBottom);
      context.stroke();
    }

    const finite = [];
    scaleSources.forEach((source) => channel[source].forEach((value) => {
      if (value != null && Number.isFinite(value)) finite.push(value);
    }));
    let minimum = finite.length ? Math.min(...finite) : -1;
    let maximumValue = finite.length ? Math.max(...finite) : 1;
    if (minimum === maximumValue) {
      minimum -= 1;
      maximumValue += 1;
    }
    if (isDifference) {
      const extent = Math.max(Math.abs(minimum), Math.abs(maximumValue), 0.05) * 1.08;
      minimum = -extent;
      maximumValue = extent;
    } else {
      const padding = (maximumValue - minimum) * 0.08;
      minimum -= padding;
      maximumValue += padding;
    }

    if (minimum < 0 && maximumValue > 0) {
      const zeroY = mapY(0, minimum, maximumValue, rowTop, rowBottom);
      context.strokeStyle = "rgba(137,160,174,0.18)";
      context.beginPath();
      context.moveTo(left, zeroY);
      context.lineTo(left + plotWidth, zeroY);
      context.stroke();
    }

    context.fillStyle = channel.shorted ? "#ff6f7d" : "#dce8ed";
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(channel.name, left - 10, rowTop + 18);
    context.fillStyle = "#5f7888";
    context.fillText(`#${channel.index}`, left - 10, rowTop + 34);
    context.fillText(`${formatCompact(maximumValue)} µV`, left - 10, rowTop + 52);
    context.fillText(`${formatCompact(minimum)} µV`, left - 10, rowBottom - 8);

    visibleSources.forEach((source) => {
      const values = channel[source];
      context.strokeStyle = source === "erd" ? "#47d7e8" : source === "edf" ? "#ffb454" : "#c78cff";
      context.globalAlpha = visibleSources.length > 1 ? 0.86 : 0.96;
      context.lineWidth = source === "erd" ? 1.15 : 1;
      context.beginPath();
      let drawing = false;
      values.forEach((value, index) => {
        if (value == null || !Number.isFinite(value)) {
          drawing = false;
          return;
        }
        const x = left + (plotWidth * index) / Math.max(1, values.length - 1);
        const y = mapY(value, minimum, maximumValue, rowTop + 3, rowBottom - 3);
        if (!drawing) {
          context.moveTo(x, y);
          drawing = true;
        } else {
          context.lineTo(x, y);
        }
      });
      context.stroke();
      context.globalAlpha = 1;
    });
  });

  context.fillStyle = "#5f7888";
  context.textBaseline = "alphabetic";
  for (let tick = 0; tick <= 4; tick += 1) {
    const x = left + (plotWidth * tick) / 4;
    const seconds = data.startSeconds + (data.durationSeconds * tick) / 4;
    context.textAlign = tick === 0 ? "left" : tick === 4 ? "right" : "center";
    context.fillText(`${seconds.toFixed(3)} s`, x, cssHeight - 8);
  }

  data.events.forEach((event) => {
    const fraction = event.offsetSeconds / data.durationSeconds;
    const x = left + plotWidth * fraction;
    context.strokeStyle = "rgba(255,111,125,0.48)";
    context.setLineDash([3, 4]);
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x, cssHeight - bottom);
    context.stroke();
    context.setLineDash([]);
  });
}

function mapY(value, minimum, maximum, top, bottom) {
  return bottom - ((value - minimum) / (maximum - minimum)) * (bottom - top);
}

function setConnection(ready, text) {
  elements.connectionStatus.classList.toggle("ready", ready);
  elements.connectionStatus.lastElementChild.textContent = text;
}

function showError(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => { elements.toast.hidden = true; }, 5200);
}

function formatClock(seconds) {
  const rounded = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  return [hours, minutes, secs].map((value) => String(value).padStart(2, "0")).join(":");
}

function formatNumber(value, digits) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatCompact(value) {
  const absolute = Math.abs(value);
  if (absolute >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (absolute >= 10) return value.toFixed(0);
  return value.toFixed(1);
}

function maximum(values) {
  const finite = values.filter((value) => value != null && Number.isFinite(value));
  return finite.length ? Math.max(...finite) : null;
}

function clamp(value, minimum, maximumValue) {
  return Math.min(maximumValue, Math.max(minimum, value));
}
