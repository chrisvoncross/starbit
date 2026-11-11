// timeframe_switcher.js — CENTRAL, reusable Timeframe-Swap element for Horizon.
//
// Renders every interval Nexus can produce — Sec15, Min1, Min5, Min15 (Sec15
// was previously missing from the interval surface) — as a segmented toggle.
// Each interval reflects the backend's on-disk availability probe: an interval
// whose enriched-features gold dataset is NOT materialized is disabled (honest,
// never a dead click), the ACTIVE bound interval is highlighted, and the
// interval carrying the live as-of pin is badged so the operator can see which
// timeframe is the mathematically-guaranteed live training surface.
//
// Usage:
//   const tf = createTimeframeSwitcher({ onSelect: (interval) => controller.setInterval(interval) });
//   container.appendChild(tf.el);
//   tf.setIntervals(intervalsFromBackend, activeInterval);

export function createTimeframeSwitcher({ onSelect } = {}) {
    const el = document.createElement('div');
    el.className = 'hz-switcher hz-timeframe-switcher';

    const label = document.createElement('span');
    label.className = 'hz-switcher-label';
    label.textContent = 'timeframe';
    el.appendChild(label);

    const seg = document.createElement('div');
    seg.className = 'hz-tf-seg';
    el.appendChild(seg);

    let _active = null;

    function setIntervals(intervals, active) {
        _active = active;
        seg.innerHTML = '';
        for (const it of intervals || []) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'hz-tf-btn';
            btn.textContent = it.interval;
            btn.dataset.interval = it.interval;
            if (it.is_active || it.interval === active) btn.classList.add('active');
            if (!it.available) {
                btn.classList.add('unavailable');
                btn.disabled = true;
                btn.title = `${it.interval}: no gold dataset materialized`;
            } else {
                btn.title = it.has_live_pin
                    ? `${it.interval}: LIVE pinned training surface`
                    : `${it.interval}: available (latest, not live-pinned)`;
            }
            if (it.has_live_pin) {
                const dot = document.createElement('span');
                dot.className = 'hz-tf-livedot';
                dot.title = 'live pinned surface';
                btn.appendChild(dot);
            }
            btn.addEventListener('click', () => {
                if (btn.disabled || it.interval === _active) return;
                if (typeof onSelect === 'function') onSelect(it.interval);
            });
            seg.appendChild(btn);
        }
    }

    return {
        el,
        setIntervals,
        /** Highlight the active interval after a swap completes. */
        setActive(active) {
            _active = active;
            for (const btn of seg.querySelectorAll('.hz-tf-btn')) {
                btn.classList.toggle('active', btn.dataset.interval === active);
            }
        },
    };
}
