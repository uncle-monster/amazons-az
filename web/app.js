let STATE = null;
let BUSY = false;

// selection: from -> to -> arrow
let selection = {
    from: null,
    to: null,
    arrow: null,
};

const THEME_KEY = "amazons_theme"; // "light" | "dark" | "system"

function keyOf(pos) {
    return `${pos.r},${pos.c}`;
}

function getCellValue(pos) {
    return STATE.board[pos.r][pos.c];
}

function setError(msg) {
    document.getElementById("error").textContent = msg || "";
}

function setHint(msg) {
    document.getElementById("hint").textContent = msg || "";
}

function clearSelection() {
    selection = { from: null, to: null, arrow: null };
    setError("");
    updateSelectionUI();
    renderBoard();
}

function undoSelection() {
    setError("");
    if (selection.arrow) selection.arrow = null;
    else if (selection.to) selection.to = null;
    else if (selection.from) selection.from = null;
    updateSelectionUI();
    renderBoard();
}

function updateSelectionUI() {
    const el = document.getElementById("selection");
    el.textContent =
        `from=${selection.from ? keyOf(selection.from) : "-"}  ` +
        `to=${selection.to ? keyOf(selection.to) : "-"}  ` +
        `arrow=${selection.arrow ? keyOf(selection.arrow) : "-"}`;

    if (BUSY) {
        setHint("Submitting move...");
        return;
    }

    if (!selection.from) setHint("Step 1: click a P1 amazon as FROM.");
    else if (!selection.to) setHint("Step 2: click an EMPTY cell as TO (UI checks empty only).");
    else if (!selection.arrow) setHint("Step 3: click an EMPTY cell as ARROW (UI checks empty only).");
    else setHint("Selection complete.");
}

async function fetchState() {
    const res = await fetch("/state");
    STATE = await res.json();
    if (STATE.game_over) {
        document.getElementById("turnInfo").textContent =
            `Game Over. Winner: ${STATE.winner === 1 ? "Human(P1)" : "AI(P2)"}`;
    } else {
        document.getElementById("turnInfo").textContent =
            `Turn: ${STATE.turn} (1=Human, 2=AI)`;
    }
}

function renderBoard() {
    if (!STATE) return;

    const boardEl = document.getElementById("board");
    boardEl.innerHTML = "";

    const selFrom = selection.from ? keyOf(selection.from) : null;
    const selTo = selection.to ? keyOf(selection.to) : null;
    const selArrow = selection.arrow ? keyOf(selection.arrow) : null;

    for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
            const v = STATE.board[r][c];

            const cell = document.createElement("div");
            cell.className = "cell " + (((r + c) % 2 === 0) ? "light" : "dark");
            cell.dataset.key = `${r},${c}`;

            if (v === 1) cell.classList.add("p1");
            if (v === 2) cell.classList.add("p2");
            if (v === 3) cell.classList.add("block");

            if (cell.dataset.key === selFrom) cell.classList.add("sel-from");
            if (cell.dataset.key === selTo) cell.classList.add("sel-to");
            if (cell.dataset.key === selArrow) cell.classList.add("sel-arrow");

            cell.textContent = "";
            cell.addEventListener("click", () => onCellClick({ r, c }));

            boardEl.appendChild(cell);
        }
    }
}

async function submitMoveIfReady() {
    if (!(selection.from && selection.to && selection.arrow)) return;

    BUSY = true;
    updateSelectionUI();

    const payload = {
        from_pos: selection.from,
        to_pos: selection.to,
        arrow_pos: selection.arrow,
    };

    try {
        const res = await fetch("/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const text = await res.text();
            throw new Error(text);
        }

        STATE = await res.json();
        document.getElementById("turnInfo").textContent = `Turn: ${STATE.turn} (1=Human, 2=AI)`;

        // After human move succeeds, if it's AI turn, ask AI to move
        if (STATE.turn === 2) {
            BUSY = true;
            updateSelectionUI();
            try {
                const aiRes = await fetch("/ai_move?simulations=200", { method: "POST" });
                if (!aiRes.ok) {
                    const text = await aiRes.text();
                    throw new Error(text);
                }
                STATE = await aiRes.json();
                document.getElementById("turnInfo").textContent = `Turn: ${STATE.turn} (1=Human, 2=AI)`;
                renderBoard();
            } catch (e) {
                setError(`AI move failed: ${e.message}`);
            } finally {
                BUSY = false;
                updateSelectionUI();
            }
        }

        selection = { from: null, to: null, arrow: null };
        setError("");
        renderBoard();
        updateSelectionUI();
    } catch (e) {
        setError(`Move rejected by server: ${e.message}`);
        updateSelectionUI();
    } finally {
        BUSY = false;
        updateSelectionUI();
    }
}

function onCellClick(pos) {
    if (!STATE || BUSY) return;
    if (STATE.game_over) {
        setError("Game is over. Click Reset to start a new game.");
        return;
    }

    setError("");

    // If selection is complete, start over (but still must be P1)
    if (selection.from && selection.to && selection.arrow) {
        selection = { from: null, to: null, arrow: null };
    }

    // Only allow human to select when it's human turn
    if (STATE.turn !== 1) {
        setError("Not your turn (turn != 1).");
        return;
    }

    const v = getCellValue(pos);

    if (!selection.from) {
        if (v !== 1) {
            setError(`FROM must be a P1 amazon. You clicked value=${v} at ${keyOf(pos)}.`);
            return;
        }
        selection.from = pos;
    } else if (!selection.to) {
        if (v !== 0) {
            setError(`TO must be an EMPTY cell. You clicked value=${v} at ${keyOf(pos)}.`);
            return;
        }
        if (selection.from.r === pos.r && selection.from.c === pos.c) {
            setError("TO cannot be the same as FROM.");
            return;
        }
        selection.to = pos;
    } else if (!selection.arrow) {
        const isFromSquare = selection.from && selection.from.r === pos.r && selection.from.c === pos.c;

        if (v !== 0 && !isFromSquare) {
            setError(`ARROW must be an EMPTY cell (or the original FROM square). You clicked value=${v} at ${keyOf(pos)}.`);
            return;
        }

        if (selection.to && selection.to.r === pos.r && selection.to.c === pos.c) {
            setError("ARROW cannot be the same as TO.");
            return;
        }

        selection.arrow = pos;
    }

    updateSelectionUI();
    renderBoard();
    submitMoveIfReady();
}

async function resetGame() {
    await fetch("/reset", { method: "POST" });
    await fetchState();
    clearSelection();
}

/* ---------------- Theme (System / Light / Dark) ---------------- */

function getSystemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
}

function applyTheme(choice) {
    // choice: "system" | "light" | "dark"
    const html = document.documentElement;

    if (choice === "system") {
        html.setAttribute("data-theme", getSystemTheme());
    } else {
        html.setAttribute("data-theme", choice);
    }
}

function initTheme() {
    const select = document.getElementById("themeSelect");
    if (!select) return;

    const saved = localStorage.getItem(THEME_KEY) || "system";
    select.value = saved;
    applyTheme(saved);

    select.addEventListener("change", () => {
        const choice = select.value;
        localStorage.setItem(THEME_KEY, choice);
        applyTheme(choice);
    });

    // react to OS theme changes if System selected
    if (window.matchMedia) {
        const mq = window.matchMedia("(prefers-color-scheme: dark)");
        const handler = () => {
            if ((localStorage.getItem(THEME_KEY) || "system") === "system") {
                applyTheme("system");
            }
        };
        if (mq.addEventListener) mq.addEventListener("change", handler);
        else mq.addListener(handler);
    }
}

/* -------------------------------------------------------------- */

async function main() {
    document.getElementById("resetBtn").addEventListener("click", resetGame);
    document.getElementById("undoBtn").addEventListener("click", undoSelection);
    document.getElementById("clearBtn").addEventListener("click", clearSelection);

    initTheme();

    await fetchState();
    clearSelection();
}

main();