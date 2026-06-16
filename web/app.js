const teams = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
  "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
  "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
];

const scenarios = {
  example: {
    away_team: "NYG",
    home_team: "WAS",
    spread_line: 1.5,
    total_line: 41.5,
    away_moneyline: 115,
    home_moneyline: -135,
    away_rest: 6,
    home_rest: 7,
    div_game: 1,
    roof: "outdoors",
    surface: "grass",
    temp: 65,
    wind: 8
  },
  closeGame: {
    away_team: "BAL",
    home_team: "CIN",
    spread_line: 1.0,
    total_line: 44.5,
    away_moneyline: -105,
    home_moneyline: -115,
    away_rest: 7,
    home_rest: 7,
    div_game: 1,
    roof: "outdoors",
    surface: "fieldturf",
    temp: 52,
    wind: 10
  },
  homeUnderdog: {
    away_team: "KC",
    home_team: "DEN",
    spread_line: -4.5,
    total_line: 45.0,
    away_moneyline: -205,
    home_moneyline: 175,
    away_rest: 7,
    home_rest: 6,
    div_game: 1,
    roof: "outdoors",
    surface: "grass",
    temp: 38,
    wind: 12
  },
  weatherGame: {
    away_team: "BUF",
    home_team: "CLE",
    spread_line: 2.0,
    total_line: 39.5,
    away_moneyline: 120,
    home_moneyline: -140,
    away_rest: 7,
    home_rest: 7,
    div_game: 0,
    roof: "outdoors",
    surface: "grass",
    temp: 28,
    wind: 22
  }
};

const featureDescriptions = {
  spread_line: "Spread line",
  total_line: "Game total",
  rest_diff: "Home rest edge",
  home_implied_prob: "Home market probability",
  implied_prob_diff: "Market probability edge",
  div_game: "Division game",
  is_dome: "Dome or closed roof",
  is_turf: "Artificial surface",
  temp: "Temperature",
  wind: "Wind",
  home_win_pct: "Home win rate",
  away_win_pct: "Away win rate",
  win_pct_diff: "Win rate edge",
  home_points_for_avg: "Home scoring average",
  away_points_for_avg: "Away scoring average",
  points_for_diff: "Scoring edge",
  home_points_allowed_avg: "Home points allowed",
  away_points_allowed_avg: "Away points allowed",
  points_allowed_diff: "Defense edge",
  home_point_diff_avg: "Home point margin",
  away_point_diff_avg: "Away point margin",
  point_diff_diff: "Point margin edge",
  games_played_diff: "Experience edge"
};

const importanceDescriptions = {
  spread_line: "Spread line",
  home_implied_prob: "Home market probability",
  implied_prob_diff: "Market probability edge",
  is_turf: "Artificial surface",
  is_dome: "Dome or closed roof",
  total_line: "Game total",
  div_game: "Division game",
  temp: "Temperature",
  wind: "Wind",
  rest_diff: "Rest edge",
  home_win_pct: "Home win rate",
  away_win_pct: "Away win rate",
  win_pct_diff: "Win rate edge",
  home_points_for_avg: "Home scoring average",
  away_points_for_avg: "Away scoring average",
  points_for_diff: "Scoring edge",
  home_points_allowed_avg: "Home points allowed",
  away_points_allowed_avg: "Away points allowed",
  points_allowed_diff: "Defense edge",
  home_point_diff_avg: "Home point margin",
  away_point_diff_avg: "Away point margin",
  point_diff_diff: "Point margin edge",
  games_played_diff: "Experience edge"
};

const metricDescriptions = {
  "Best model": "Algorithm selected by ROC AUC.",
  "Rows": "Total real NFL games in the dataset.",
  "Train rows": "Older games used to fit the model.",
  "Test rows": "Later games held out for evaluation.",
  "Accuracy": "Share of test games picked correctly.",
  "ROC AUC": "How well the model ranks likely winners.",
  "Brier score": "Probability error; lower is better.",
  "Log loss": "Confidence penalty; lower is better."
};

const form = document.querySelector("#prediction-form");

function currentMode() {
  return form.elements.model_mode.value;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function populateTeams() {
  for (const select of document.querySelectorAll("select[name='away_team'], select[name='home_team']")) {
    select.innerHTML = teams.map((team) => `<option value="${team}">${team}</option>`).join("");
  }
  setFormValues(scenarios.example);
}

function setFormValues(values) {
  for (const [name, value] of Object.entries(values)) {
    const field = form.elements[name];
    if (!field) continue;
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
    } else {
      field.value = value;
    }
  }
}

function readForm() {
  const data = new FormData(form);
  const numericFields = [
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "home_moneyline",
    "away_moneyline",
    "temp",
    "wind"
  ];
  const payload = {
    model_mode: currentMode(),
    home_team: data.get("home_team"),
    away_team: data.get("away_team"),
    roof: data.get("roof"),
    surface: data.get("surface"),
    div_game: form.elements.div_game.checked ? 1 : 0
  };
  for (const field of numericFields) {
    payload[field] = Number(data.get(field));
  }
  return payload;
}

function updateModeUi() {
  const isTeamMode = currentMode() === "team";
  document.querySelectorAll(".market-field").forEach((field) => {
    field.hidden = isTeamMode;
  });
}

function modeLabel(mode) {
  if (mode === "team") return "Team-stat";
  if (mode === "combined") return "Combined";
  return "Market";
}

function renderPrediction(result) {
  const homePct = formatPercent(result.home_win_probability);
  const awayPct = formatPercent(result.away_win_probability);
  const pickProbability = result.pick === result.home_team
    ? result.home_win_probability
    : result.away_win_probability;
  const edge = Math.abs(result.home_win_probability - 0.5);
  const edgeText = edge < 0.06 ? "nearly even" : edge < 0.12 ? "slightly favored" : "favored";

  document.querySelector("#result-matchup").textContent = result.matchup;
  document.querySelector("#confidence").textContent =
    `${modeLabel(result.model_mode)} | ${result.confidence} confidence`;
  document.querySelector("#pick").textContent = result.pick;
  document.querySelector("#prediction-summary").textContent =
    `${result.pick} is ${edgeText} with a ${formatPercent(pickProbability)} win probability. ` +
    `This is a ${result.confidence.toLowerCase()}-confidence model edge.`;
  document.querySelector("#home-probability").textContent = homePct;
  document.querySelector("#home-ring").style.background =
    `conic-gradient(var(--green) ${result.home_win_probability * 100}%, #e3e7e2 0)`;
  document.querySelector("#home-label").textContent = `${result.home_team} win`;
  document.querySelector("#away-label").textContent = `${result.away_team} win`;
  document.querySelector("#home-bar-value").textContent = homePct;
  document.querySelector("#away-bar-value").textContent = awayPct;
  document.querySelector("#home-bar").style.width = homePct;
  document.querySelector("#away-bar").style.width = awayPct;

  const featureBox = document.querySelector("#feature-values");
  featureBox.innerHTML = Object.entries(result.features)
    .map(([key, value]) => `
      <div class="feature-pill">
        ${featureDescriptions[key] || key}
        <strong>${value}</strong>
        <span>${key}</span>
      </div>
    `)
    .join("");
}

async function predict() {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readForm())
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text);
  }
  renderPrediction(await response.json());
}

async function loadMetrics() {
  const response = await fetch(`/api/metrics?model_mode=${currentMode()}`);
  const metrics = await response.json();
  const selected = metrics.models[metrics.best_model];
  document.querySelector("#rows").textContent = metrics.rows.toLocaleString();
  document.querySelector("#auc").textContent = selected.roc_auc.toFixed(3);
  document.querySelector("#accuracy").textContent = formatPercent(selected.accuracy);
  document.querySelector("#metrics-table").innerHTML = [
    ["Best model", metrics.best_model.replace("_", " ")],
    ["Rows", metrics.rows.toLocaleString()],
    ["Train rows", metrics.train_rows.toLocaleString()],
    ["Test rows", metrics.test_rows.toLocaleString()],
    ["Accuracy", formatPercent(selected.accuracy)],
    ["ROC AUC", selected.roc_auc.toFixed(4)],
    ["Brier score", selected.brier_score.toFixed(4)],
    ["Log loss", selected.log_loss.toFixed(4)]
  ]
    .map(([label, value]) => `
      <div class="metric-card">
        <small>${label}</small>
        <strong>${value}</strong>
        <span>${metricDescriptions[label]}</span>
      </div>
    `)
    .join("");
}

async function loadImportance() {
  const response = await fetch(`/api/feature-importance?model_mode=${currentMode()}`);
  const rows = await response.json();
  const topRows = rows.slice(0, 8);
  const max = Math.max(...topRows.map((row) => Math.abs(row.importance)));
  document.querySelector("#importance-chart").innerHTML = topRows
    .map((row) => {
      const width = `${Math.max((Math.abs(row.importance) / max) * 100, 4)}%`;
      return `
        <div class="importance-row">
          <strong>${importanceDescriptions[row.feature] || row.feature}</strong>
          <div class="importance-bar-track"><div class="importance-bar" style="width: ${width}"></div></div>
          <span>${row.importance.toFixed(3)}</span>
        </div>
      `;
    })
    .join("");
}

document.querySelectorAll("[data-scenario]").forEach((button) => {
  button.addEventListener("click", () => {
    setFormValues(scenarios[button.dataset.scenario]);
    predict().catch(console.error);
  });
});

document.querySelectorAll("input[name='model_mode']").forEach((input) => {
  input.addEventListener("change", () => {
    updateModeUi();
    loadMetrics().catch(console.error);
    loadImportance().catch(console.error);
    predict().catch(console.error);
  });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  predict().catch((error) => {
    alert("Prediction failed. Check your inputs and make sure the model has been trained.");
    console.error(error);
  });
});

populateTeams();
updateModeUi();
loadMetrics().catch(console.error);
loadImportance().catch(console.error);
predict().catch(console.error);
