(() => {
  const PAIRS = {
    BTCUSD: { label: "BTC / USD", base: 68420, vol: 180 },
    ETHUSD: { label: "ETH / USD", base: 3420, vol: 22 },
    SOLUSD: { label: "SOL / USD", base: 148.5, vol: 1.8 },
  };

  const pairLabelEl = document.getElementById("pairLabel");
  const priceEl = document.getElementById("price");
  const deltaEl = document.getElementById("delta");
  const deltaAbsEl = document.getElementById("deltaAbs");
  const deltaPctEl = document.getElementById("deltaPct");
  const statusTextEl = document.getElementById("statusText");
  const chartEl = document.getElementById("chart");

  let activePair = "BTCUSD";
  let intervalMin = 5;
  let series = null;
  let chart = null;
  let candles = [];
  let sessionOpen = 0;
  let tickTimer = null;
  let clock = Math.floor(Date.now() / 1000);

  function formatPrice(value) {
    if (value >= 1000) {
      return value.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });
    }
    return value.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    });
  }

  function formatSigned(value, digits = 2) {
    const sign = value >= 0 ? "+" : "−";
    return `${sign}${Math.abs(value).toFixed(digits)}`;
  }

  function seededNoise(seed) {
    const x = Math.sin(seed * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  }

  function buildHistory(pairKey, minutes, bars = 180) {
    const cfg = PAIRS[pairKey];
    const step = minutes * 60;
    let t = clock - bars * step;
    let price = cfg.base * (0.96 + seededNoise(bars) * 0.08);
    const out = [];

    for (let i = 0; i < bars; i += 1) {
      const drift = Math.sin(i / 11) * cfg.vol * 0.15;
      const shock = (seededNoise(i + 3.7) - 0.48) * cfg.vol;
      const open = price;
      const close = Math.max(0.01, open + drift + shock);
      const high = Math.max(open, close) + seededNoise(i + 1.1) * cfg.vol * 0.35;
      const low = Math.min(open, close) - seededNoise(i + 2.2) * cfg.vol * 0.35;
      out.push({
        time: t,
        open: +open.toFixed(4),
        high: +high.toFixed(4),
        low: +low.toFixed(4),
        close: +close.toFixed(4),
      });
      price = close;
      t += step;
    }
    return out;
  }

  function updateQuote() {
    if (!candles.length) return;
    const last = candles[candles.length - 1];
    const abs = last.close - sessionOpen;
    const pct = (abs / sessionOpen) * 100;

    priceEl.textContent = formatPrice(last.close);
    deltaAbsEl.textContent = formatSigned(abs, last.close >= 1000 ? 2 : 3);
    deltaPctEl.textContent = `${formatSigned(pct, 2)}%`;
    deltaEl.classList.toggle("up", abs >= 0);
    deltaEl.classList.toggle("down", abs < 0);

    priceEl.classList.remove("flash-up", "flash-down");
    void priceEl.offsetWidth;
    priceEl.classList.add(abs >= 0 ? "flash-up" : "flash-down");
  }

  function createChart() {
    if (chart) {
      chart.remove();
      chart = null;
      series = null;
    }

    chart = LightweightCharts.createChart(chartEl, {
      layout: {
        background: { type: "solid", color: "transparent" },
        textColor: "#6a6a6a",
        fontFamily: "Syne, Helvetica Neue, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.03)" },
        horzLines: { color: "rgba(255,255,255,0.03)" },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {
          color: "rgba(200,196,188,0.25)",
          width: 1,
          style: LightweightCharts.LineStyle.SparseDotted,
          labelBackgroundColor: "#1a1a1a",
        },
        horzLine: {
          color: "rgba(200,196,188,0.25)",
          width: 1,
          style: LightweightCharts.LineStyle.SparseDotted,
          labelBackgroundColor: "#1a1a1a",
        },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });

    series = chart.addCandlestickSeries({
      upColor: "#b8b2a6",
      downColor: "#3a3a36",
      borderUpColor: "#d4d0c6",
      borderDownColor: "#2a2a28",
      wickUpColor: "#9a968c",
      wickDownColor: "#3a3a36",
    });

    chart.timeScale().fitContent();
  }

  function loadPair(pairKey) {
    activePair = pairKey;
    pairLabelEl.textContent = PAIRS[pairKey].label;
    candles = buildHistory(pairKey, intervalMin);
    sessionOpen = candles[0].open;
    series.setData(candles);
    chart.timeScale().fitContent();
    updateQuote();
    statusTextEl.textContent = "live feed";
  }

  function tick() {
    if (!candles.length) return;
    const cfg = PAIRS[activePair];
    const step = intervalMin * 60;
    const now = Math.floor(Date.now() / 1000);
    clock = now;

    const last = candles[candles.length - 1];
    const bucket = Math.floor(now / step) * step;
    const nudge = (Math.random() - 0.5) * cfg.vol * 0.22;
    const nextClose = Math.max(0.01, last.close + nudge);

    if (bucket > last.time) {
      const next = {
        time: bucket,
        open: last.close,
        high: Math.max(last.close, nextClose),
        low: Math.min(last.close, nextClose),
        close: +nextClose.toFixed(4),
      };
      candles.push(next);
      if (candles.length > 220) candles.shift();
      series.update(next);
    } else {
      last.close = +nextClose.toFixed(4);
      last.high = Math.max(last.high, last.close);
      last.low = Math.min(last.low, last.close);
      series.update(last);
    }

    updateQuote();
  }

  function bindUi() {
    document.querySelectorAll(".pair").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".pair").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        loadPair(btn.dataset.pair);
      });
    });

    document.querySelectorAll(".interval").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".interval").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        intervalMin = Number(btn.dataset.interval);
        loadPair(activePair);
      });
    });

    window.addEventListener("resize", () => {
      if (!chart) return;
      chart.applyOptions({
        width: chartEl.clientWidth,
        height: chartEl.clientHeight,
      });
    });
  }

  function start() {
    createChart();
    chart.applyOptions({
      width: chartEl.clientWidth,
      height: chartEl.clientHeight,
    });
    bindUi();
    loadPair(activePair);
    tickTimer = window.setInterval(tick, 900);
  }

  start();
})();
