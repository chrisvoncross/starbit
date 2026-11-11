/**
 * Z-Score Probability Table Overlay
 * 
 * Displays a reference table showing Z-Score values and their corresponding
 * cumulative probabilities from the standard normal distribution.
 * Highlights the current Z-Score range.
 */

/**
 * Initialize Z-Score Table
 * Loads the HTML and CSS, injects into the page
 */
export async function initZScoreTable() {
    // Load CSS
    const cssLink = document.createElement('link');
    cssLink.rel = 'stylesheet';
    cssLink.href = '/indicators/zscore/zscore-table.css';
    document.head.appendChild(cssLink);

    // Load HTML
    const response = await fetch('/indicators/zscore/zscore-table.html');
    const html = await response.text();

    // Inject into body
    document.body.insertAdjacentHTML('beforeend', html);
}

/**
 * Update Z-Score Table highlight based on current Z-Score value
 */
export function updateZScoreTable(zscoreValue) {
    if (zscoreValue === undefined || zscoreValue === null) return;
    
    // Get all Z-Score cells
    const zscoreCells = document.querySelectorAll('#zscore-table .zscore-cell');
    
    // Remove all active states
    zscoreCells.forEach(cell => cell.classList.remove('active'));
    
    // Determine which cell to highlight
    let targetZScore;
    if (zscoreValue >= 3) {
        targetZScore = '3';
    } else if (zscoreValue >= 2) {
        targetZScore = '2';
    } else if (zscoreValue >= 1) {
        targetZScore = '1';
    } else if (zscoreValue >= 0) {
        targetZScore = '0';
    } else if (zscoreValue >= -1) {
        targetZScore = '-1';
    } else if (zscoreValue >= -2) {
        targetZScore = '-2';
    } else {
        targetZScore = '-3';
    }
    
    // Highlight the appropriate cell
    const targetCell = document.querySelector(`#zscore-table .zscore-cell[data-zscore="${targetZScore}"]`);
    if (targetCell) {
        targetCell.classList.add('active');
    }
}

/**
 * Show Z-Score Table (called after pane is created)
 */
export function showZScoreTable() {
    const zscoreTable = document.getElementById('zscore-table');
    if (zscoreTable) {
        zscoreTable.style.display = 'block';
    }
}

