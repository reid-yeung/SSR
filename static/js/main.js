(function () {
  // ---------- 左侧阅读进度导航 ----------
  const pnav = document.getElementById("pnav");
  if (pnav) {
    const links = Array.from(pnav.querySelectorAll("a"));
    const sections = links.map((a) => document.querySelector(a.getAttribute("href")));
    const fill = pnav.querySelector(".pnav-fill");
    const hero = document.querySelector(".hero");

    // 每个圆点按位置从 --c-a 渐变到 --c-b
    const css = getComputedStyle(document.documentElement);
    const ca = hexToRgb(css.getPropertyValue("--c-a"));
    const cb = hexToRgb(css.getPropertyValue("--c-b"));
    links.forEach((a, i) => {
      const t = links.length > 1 ? i / (links.length - 1) : 0;
      const c = ca.map((v, k) => Math.round(v + (cb[k] - v) * t));
      a.style.setProperty("--c", `rgb(${c.join(",")})`);
    });

    function update() {
      const probe = window.scrollY + window.innerHeight * 0.3;
      const tops = sections.map((s) => s.getBoundingClientRect().top + window.scrollY);
      let i = 0;
      while (i + 1 < tops.length && tops[i + 1] <= probe) i++;
      const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
      if (atBottom) i = links.length - 1;

      // 进度条在当前圆点与下一个圆点之间按滚动比例延伸
      let frac = 0;
      if (!atBottom && i + 1 < tops.length) {
        frac = Math.min(1, Math.max(0, (probe - tops[i]) / (tops[i + 1] - tops[i])));
      }
      const pos = (i + frac) / (links.length - 1);
      fill.style.setProperty("--hide", `${(1 - pos) * 100}%`);

      links.forEach((a, k) => {
        a.classList.toggle("active", k === i);
        a.classList.toggle("done", k < i);
      });
      pnav.classList.toggle("visible", window.scrollY > hero.offsetHeight * 0.35);
    }

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  function hexToRgb(hex) {
    const h = hex.trim().replace("#", "");
    return [0, 2, 4].map((o) => parseInt(h.slice(o, o + 2), 16));
  }

  // ---------- 深色 / 浅色切换 ----------
  // 没有手动选过时跟随系统；点按钮后写入 data-theme 并记住选择
  const root = document.documentElement;
  const themeBtn = document.getElementById("theme-btn");
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  const isDark = () => root.getAttribute("data-theme") === "dark" || (!root.hasAttribute("data-theme") && systemDark.matches);

  function syncThemeBtn() {
    const dark = isDark();
    // SVG 元素没有 hidden 属性的 JS 映射，只能直接改特性
    themeBtn.querySelector(".i-moon").toggleAttribute("hidden", dark);
    themeBtn.querySelector(".i-sun").toggleAttribute("hidden", !dark);
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }

  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = isDark() ? "light" : "dark";
      // 切换时给颜色一个短暂的过渡
      root.classList.add("theme-anim");
      root.setAttribute("data-theme", next);
      try {
        localStorage.setItem("theme", next);
      } catch (e) {}
      syncThemeBtn();
      setTimeout(() => root.classList.remove("theme-anim"), 400);
    });
    systemDark.addEventListener("change", syncThemeBtn);
    syncThemeBtn();
  }

  // ---------- BibTeX 一键复制 ----------
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    const label = btn.querySelector("span");
    btn.addEventListener("click", async () => {
      const text = document.getElementById(btn.dataset.copyTarget).innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (err) {
        // 非 HTTPS 环境下 clipboard API 不可用，退回旧方法
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
      }
      btn.classList.add("copied");
      label.textContent = "Copied!";
      setTimeout(() => {
        btn.classList.remove("copied");
        label.textContent = "Copy";
      }, 1600);
    });
  });

  // ---------- 选中文字的荧光笔效果 ----------
  // 浏览器原生的选区只能是纯色，这里按行画出渐变色块，每一行从左往右扫入。
  // 只有脚本正常运行后才隐藏原生选区（html.hl-ready），否则仍是默认的蓝色选区。
  const layer = document.createElement("div");
  layer.className = "hl-layer";
  layer.setAttribute("aria-hidden", "true");
  document.body.appendChild(layer);
  document.documentElement.classList.add("hl-ready");

  const marks = [];
  let queued = false;

  // 文字所在的可滚动 / 裁切容器（代码块、表格）的可见区域；超出部分的文字看不到，色块也不能画出去
  function visibleBox(el) {
    let box = { left: -Infinity, top: -Infinity, right: Infinity, bottom: Infinity };
    for (; el && el !== document.body; el = el.parentElement) {
      const cs = getComputedStyle(el);
      if (cs.overflowX !== "visible" || cs.overflowY !== "visible") {
        const r = el.getBoundingClientRect();
        box = {
          left: Math.max(box.left, r.left + el.clientLeft),
          top: Math.max(box.top, r.top + el.clientTop),
          right: Math.min(box.right, r.left + el.clientLeft + el.clientWidth),
          bottom: Math.min(box.bottom, r.top + el.clientTop + el.clientHeight),
        };
      }
    }
    return box;
  }

  // 只取选区里文字的矩形，避免把整块元素的外框也画进去
  function textRects(range) {
    const root = range.commonAncestorContainer;
    const rects = [];
    const addNode = (node) => {
      const r = document.createRange();
      r.selectNodeContents(node);
      if (node === range.startContainer) r.setStart(node, range.startOffset);
      if (node === range.endContainer) r.setEnd(node, range.endOffset);
      const box = visibleBox(node.parentElement);
      for (const rect of r.getClientRects()) {
        const c = {
          left: Math.max(rect.left, box.left),
          right: Math.min(rect.right, box.right),
          top: Math.max(rect.top, box.top),
          bottom: Math.min(rect.bottom, box.bottom),
        };
        c.width = c.right - c.left;
        c.height = c.bottom - c.top;
        if (c.width > 1 && c.height > 1) rects.push(c);
      }
    };
    if (root.nodeType === Node.TEXT_NODE) {
      addNode(root);
      return rects;
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) =>
        n.textContent.trim() && range.intersectsNode(n) && !n.parentElement.closest(".hl-layer")
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT,
    });
    let n;
    let count = 0;
    while ((n = walker.nextNode()) && count++ < 2000) addNode(n);
    return rects;
  }

  // 同一行上的多个矩形（含上标）合并成一条：纵向重叠超过较矮者的三成就算同一行
  function sameLine(a, b) {
    const overlap = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    return overlap > Math.min(a.bottom - a.top, b.bottom - b.top) * 0.3;
  }

  function mergeLines(rects) {
    let lines = rects.map((r) => ({ left: r.left, right: r.right, top: r.top, bottom: r.bottom }));
    let merged = true;
    while (merged) {
      merged = false;
      const out = [];
      for (const r of lines) {
        const line = out.find((l) => sameLine(l, r));
        if (line) {
          line.left = Math.min(line.left, r.left);
          line.right = Math.max(line.right, r.right);
          line.top = Math.min(line.top, r.top);
          line.bottom = Math.max(line.bottom, r.bottom);
          merged = true;
        } else {
          out.push({ ...r });
        }
      }
      lines = out;
    }
    return lines.sort((a, b) => a.top - b.top);
  }

  function render() {
    queued = false;
    const sel = window.getSelection();
    let lines = [];
    if (sel && sel.rangeCount && !sel.isCollapsed) {
      for (let i = 0; i < sel.rangeCount; i++) lines = lines.concat(mergeLines(textRects(sel.getRangeAt(i))));
    }
    const sx = window.scrollX;
    const sy = window.scrollY;
    lines.forEach((l, i) => {
      let m = marks[i];
      if (!m) {
        // 新出现的一行才播放扫入动画，拖动过程中已有的行只更新位置
        m = document.createElement("i");
        m.style.animationDelay = `${Math.min(i, 8) * 40}ms`;
        layer.appendChild(m);
        marks.push(m);
      }
      const pad = (l.bottom - l.top) * 0.08;
      m.style.left = `${l.left + sx - 2}px`;
      m.style.top = `${l.top + sy - pad}px`;
      m.style.width = `${l.right - l.left + 4}px`;
      m.style.height = `${l.bottom - l.top + pad * 2}px`;
      // 渐变按整页宽度铺开，所以不同位置的行颜色连续过渡
      m.style.backgroundPosition = `${-(l.left + sx - 2)}px 0`;
    });
    while (marks.length > lines.length) marks.pop().remove();
  }

  function queue() {
    if (!queued) {
      queued = true;
      requestAnimationFrame(render);
    }
  }

  document.addEventListener("selectionchange", queue);
  window.addEventListener("resize", queue);
  // 代码块、表格横向滚动时，色块跟着文字走（页面本身的滚动不需要重画）
  document.addEventListener(
    "scroll",
    (e) => {
      if (e.target !== document) queue();
    },
    { capture: true, passive: true }
  );
})();
