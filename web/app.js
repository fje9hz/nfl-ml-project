const teams = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
  "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
  "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
];

const example = {
  away_team: "DAL",
  home_team: "SF",
  spread_line: 3.5,
  total_line: 47.5,
  away_moneyline: 150,
  home_moneyline: -170,
  away_rest: 6,
  home_rest: 7,
  div_game: 1,
  roof: "outdoors",
  surface: "grass",
  temp: 65,
  wind: 8
};

const form = document.querySelector("#prediction-form");

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function populateTeams() {
  for (const select of document.querySelectorAll("select[name='away_team'], select[name='home_team']")) {
    select.innerHTML = teams.map((team) => `<option value="${team}">${team}</option>`).join("");
  }
  setFormValues(example);
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

function renderPrediction(result) {
  const homePct = formatPercent(result.home_win_probability);
  const awayPct = formatPercent(result.away_win_probability);

  document.querySelector("#result-matchup").textContent = result.matchup;
  document.querySelector("#confidence").textContent = `${result.confidence} confidence`;
  document.querySelector("#pick").textContent = result.pick;
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
    .map(([key, value]) => `<div class="feature-pill">${key}<strong>${value}</strong></div>`)
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
  const response = await fetch("/api/metrics");
  const metrics = await response.json();
  const logistic = metrics.models.logistic_regression;
  document.querySelector("#rows").textContent = metrics.rows.toLocaleString();
  document.querySelector("#auc").textContent = logistic.roc_auc.toFixed(3);
  document.querySelector("#accuracy").textContent = formatPercent(logistic.accuracy);
  document.querySelector("#metrics-table").innerHTML = [
    ["Best model", metrics.best_model.replace("_", " ")],
    ["Rows", metrics.rows.toLocaleString()],
    ["Train rows", metrics.train_rows.toLocaleString()],
    ["Test rows", metrics.test_rows.toLocaleString()],
    ["Accuracy", formatPercent(logistic.accuracy)],
    ["ROC AUC", logistic.roc_auc.toFixed(4)],
    ["Brier score", logistic.brier_score.toFixed(4)],
    ["Log loss", logistic.log_loss.toFixed(4)]
  ]
    .map(([label, value]) => `<div class="metric-card"><small>${label}</small><strong>${value}</strong></div>`)
    .join("");
}

async function loadImportance() {
  const response = await fetch("/api/feature-importance");
  const rows = await response.json();
  const topRows = rows.slice(0, 8);
  const max = Math.max(...topRows.map((row) => Math.abs(row.importance)));
  document.querySelector("#importance-chart").innerHTML = topRows
    .map((row) => {
      const width = `${Math.max((Math.abs(row.importance) / max) * 100, 4)}%`;
      return `
        <div class="importance-row">
          <strong>${row.feature}</strong>
          <div class="importance-bar-track"><div class="importance-bar" style="width: ${width}"></div></div>
          <span>${row.importance.toFixed(3)}</span>
        </div>
      `;
    })
    .join("");
}

document.querySelector("#load-example").addEventListener("click", () => {
  setFormValues(example);
  predict().catch(console.error);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  predict().catch((error) => {
    alert("Prediction failed. Check your inputs and make sure the model has been trained.");
    console.error(error);
  });
});

populateTeams();
loadMetrics().catch(console.error);
loadImportance().catch(console.error);
predict().catch(console.error);
