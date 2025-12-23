// Football Prediction App JavaScript

// Load example data
async function loadExample(type) {
    try {
        const response = await fetch('/examples');
        const examples = await response.json();
        const example = examples[type];

        if (!example) return;

        // Fill home team
        document.getElementById('home-name').value = example.home.name;
        fillInputs('home-scored', example.home.goals_scored);
        fillInputs('home-conceded', example.home.goals_conceded);
        fillInputs('home-first-half', example.home.first_half_goals);

        // Fill away team
        document.getElementById('away-name').value = example.away.name;
        fillInputs('away-scored', example.away.goals_scored);
        fillInputs('away-conceded', example.away.goals_conceded);
        fillInputs('away-first-half', example.away.first_half_goals);

        // Add visual feedback
        addFeedback('Example loaded! Click "Get Prediction" to see results.');
    } catch (error) {
        console.error('Error loading example:', error);
        addFeedback('Error loading example', true);
    }
}

function fillInputs(prefix, values) {
    values.forEach((val, idx) => {
        const input = document.getElementById(`${prefix}-${idx + 1}`);
        if (input) {
            input.value = val;
            // Add animation
            input.style.animation = 'none';
            setTimeout(() => {
                input.style.animation = 'fadeIn 0.3s ease-out';
            }, idx * 50);
        }
    });
}

// Get prediction
// --- Feedback Loop ---
// Store last prediction for feedback
let lastPrediction = null;

function submitFeedback() {
    const homeGoals = document.getElementById('real-home-goals').value;
    const awayGoals = document.getElementById('real-away-goals').value;
    const htHomeGoals = document.getElementById('ht-home-goals').value;
    const htAwayGoals = document.getElementById('ht-away-goals').value;
    const msgDiv = document.getElementById('feedback-message');

    if (homeGoals === '' || awayGoals === '') {
        msgDiv.textContent = "Please enter full-time scores at minimum!";
        msgDiv.className = "feedback-message error";
        return;
    }

    // Get team names from inputs
    const homeName = document.getElementById('home-name').value;
    const awayName = document.getElementById('away-name').value;

    // Prepare payload with prediction data for enhanced learning
    const payload = {
        home_team: homeName,
        away_team: awayName,
        home_goals: parseInt(homeGoals),
        away_goals: parseInt(awayGoals),
        ht_home_goals: htHomeGoals !== '' ? parseInt(htHomeGoals) : null,
        ht_away_goals: htAwayGoals !== '' ? parseInt(htAwayGoals) : null,
        date: new Date().toISOString().split('T')[0] // Today's date
    };

    // Add prediction data if available (for error tracking)
    if (lastPrediction) {
        payload.predicted_home_xg = lastPrediction.expected_goals.home;
        payload.predicted_away_xg = lastPrediction.expected_goals.away;

        // Most likely score from top prediction
        const topScore = lastPrediction.full_match_predictions[0];
        if (topScore && topScore.score) {
            const scoreParts = topScore.score.split('-');
            payload.predicted_home_score = scoreParts[0];
            payload.predicted_away_score = scoreParts[1];
        }

        // Predicted outcome (H/D/A)
        const outcomes = lastPrediction.match_outcome;
        if (outcomes.home_win >= outcomes.draw && outcomes.home_win >= outcomes.away_win) {
            payload.predicted_outcome = 'H';
        } else if (outcomes.draw >= outcomes.home_win && outcomes.draw >= outcomes.away_win) {
            payload.predicted_outcome = 'D';
        } else {
            payload.predicted_outcome = 'A';
        }

        // BTTS prediction (>50% = predicted yes)
        payload.predicted_btts = lastPrediction.both_teams_score > 50 ? 1 : 0;

        // Confidence score
        payload.confidence_score = lastPrediction.confidence_score || 50;
    }

    fetch('/submit_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                let message = "✅ <b>Saved!</b> The model will use this result to improve future predictions.";

                // Show error metrics if available
                if (data.error_metrics) {
                    const metrics = data.error_metrics;
                    message += `<br><small>`;
                    message += `xG Error: ${metrics.xg_error_home.toFixed(2)} (H), ${metrics.xg_error_away.toFixed(2)} (A) | `;
                    message += `Outcome: ${metrics.outcome_correct ? '✅' : '❌'} | `;
                    message += `BTTS: ${metrics.btts_correct ? '✅' : '❌'}`;
                    message += `</small>`;
                }

                msgDiv.innerHTML = message;
                msgDiv.className = "feedback-message success";

                // Clear inputs
                document.getElementById('real-home-goals').value = '';
                document.getElementById('real-away-goals').value = '';
            } else {
                msgDiv.textContent = "Error saving: " + data.error;
                msgDiv.className = "feedback-message error";
            }
        })
        .catch(err => {
            msgDiv.textContent = "Network error.";
            msgDiv.className = "feedback-message error";
        });
}

async function getPrediction() {
    // Validate inputs
    const homeName = document.getElementById('home-name').value.trim();
    const awayName = document.getElementById('away-name').value.trim();

    if (!homeName || !awayName) {
        addFeedback('Please enter both team names', true);
        return;
    }

    // Get neutral venue flag
    const neutralVenue = document.getElementById('neutral-venue').checked;

    // Get tournament context
    const compType = isCupMatch ? document.getElementById('comp-type').value : 'LEAGUE';
    const tournamentStage = isCupMatch ? document.getElementById('tournament-stage').value : 'GROUP';
    const homeFifa = parseInt(document.getElementById('home-fifa').value) || null;
    const awayFifa = parseInt(document.getElementById('away-fifa').value) || null;

    // Collect data
    const data = {
        neutral_venue: neutralVenue,
        is_cup: isCupMatch,
        home_team: {
            name: homeName,
            league_position: parseInt(document.getElementById('home-pos').value) || 10,
            league_size: parseInt(document.getElementById('home-total').value) || 20,
            goals_scored: getInputArray('home-scored'),
            goals_conceded: getInputArray('home-conceded'),
            first_half_goals: getInputArray('home-first-half'),
            competition_type: compType,
            tournament_stage: tournamentStage,
            fifa_ranking: homeFifa
        },
        away_team: {
            name: awayName,
            league_position: parseInt(document.getElementById('away-pos').value) || 10,
            league_size: parseInt(document.getElementById('away-total').value) || 20,
            goals_scored: getInputArray('away-scored'),
            goals_conceded: getInputArray('away-conceded'),
            first_half_goals: getInputArray('away-first-half'),
            competition_type: compType,
            tournament_stage: tournamentStage,
            fifa_ranking: awayFifa
        }
    };

    // Validate arrays
    for (let team of ['home_team', 'away_team']) {
        for (let field of ['goals_scored', 'goals_conceded', 'first_half_goals']) {
            if (data[team][field].length !== 5 || data[team][field].some(v => v === null)) {
                addFeedback('Please fill in all match data (5 matches for each field)', true);
                return;
            }
        }
    }

    // Show loading
    const btn = document.querySelector('.btn-predict');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.loading-spinner');

    btnText.style.display = 'none';
    spinner.style.display = 'inline';
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            displayResults(result);
            addFeedback('Prediction generated successfully!');
        } else {
            addFeedback(result.error || 'Prediction failed', true);
        }
    } catch (error) {
        console.error('Error:', error);
        addFeedback('Network error. Please try again.', true);
    } finally {
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
        btn.disabled = false;
    }
}

function getInputArray(prefix) {
    const values = [];
    for (let i = 1; i <= 5; i++) {
        const input = document.getElementById(`${prefix}-${i}`);
        const value = input ? parseInt(input.value) : null;
        values.push(value);
    }
    return values;
}

function displayResults(result) {
    // Store prediction for feedback
    lastPrediction = result;

    // Show results container
    const resultsContainer = document.getElementById('results');
    resultsContainer.style.display = 'block';

    // Scroll to results
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Determine if neutral venue
    const isNeutral = result.neutral_venue;
    const team1Label = isNeutral ? result.home_team : result.home_team;
    const team2Label = isNeutral ? result.away_team : result.away_team;

    // Match title
    document.getElementById('match-title').textContent =
        `${team1Label} vs ${team2Label}${isNeutral ? ' (Neutral Venue)' : ''}`;

    // Expected goals
    document.getElementById('home-xg-label').textContent = team1Label;
    document.getElementById('home-xg').textContent = result.expected_goals.home;
    document.getElementById('away-xg').textContent = result.expected_goals.away;
    document.getElementById('away-xg-label').textContent = team2Label;

    // Match outcome probabilities
    document.getElementById('outcome-win-label').textContent = isNeutral ? `${team1Label} Win` : 'Home Win';
    document.getElementById('outcome-win').textContent = `${result.match_outcome.home_win}%`;
    document.getElementById('outcome-draw').textContent = `${result.match_outcome.draw}%`;
    document.getElementById('outcome-loss-label').textContent = isNeutral ? `${team2Label} Win` : 'Away Win';
    document.getElementById('outcome-loss').textContent = `${result.match_outcome.away_win}%`;

    // First half predictions
    displayScorePredictions('first-half-predictions', result.first_half_predictions);

    // Full match predictions
    displayScorePredictions('full-match-predictions', result.full_match_predictions);

    // Total goals
    displayTotalGoals(result.total_goals);

    // Insights
    displayInsights(result.insights);

    // Both teams to score
    displayBothTeamsScore(result.both_teams_score);

    // Clean sheets
    displayCleanSheets(result.clean_sheet, team1Label, team2Label);

    // Enhanced features
    displayConfidence(result.confidence_score, result.confidence_breakdown);
    displayBettingInsights(result.betting_insights);  // New function call
    displayPredictionIntervals(result.prediction_intervals, team1Label, team2Label);
    displayTeamMomentum(result.team_insights, team1Label, team2Label);
    displayModelQuality(result.model_quality);
}

function displayBettingInsights(insights) {
    const container = document.getElementById('betting-insights');
    container.innerHTML = '';

    if (!insights || Object.keys(insights).length === 0) {
        container.innerHTML = '<p class="no-data">Betting data not available</p>';
        return;
    }

    const markets = [
        { label: 'Over 1.5 Goals', key: 'over_1_5' },
        { label: 'Over 2.5 Goals', key: 'over_2_5', highlight: true },
        { label: 'Over 3.5 Goals', key: 'over_3_5' },
        { label: 'Under 2.5 Goals', key: 'under_2_5', highlight: true }
    ];

    markets.forEach((market, idx) => {
        const prob = insights[market.key] || 0;
        const div = document.createElement('div');
        div.className = `betting-item ${market.highlight ? 'highlight' : ''}`;
        div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        // Color coding
        let colorClass = 'low-prob';
        if (prob >= 65) colorClass = 'high-prob';
        else if (prob >= 50) colorClass = 'medium-prob';

        div.innerHTML = `
            <div class="betting-label">
                <span>${market.label}</span>
                <span class="betting-prob ${colorClass}">${prob}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${prob}%; background: ${prob >= 60 ? '#22c55e' : prob >= 40 ? '#f59e0b' : '#ef4444'}"></div>
            </div>
        `;
        container.appendChild(div);
    });
}

function displayScorePredictions(containerId, predictions) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    predictions.forEach((pred, idx) => {
        const item = document.createElement('div');
        item.className = 'score-item';
        item.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        item.innerHTML = `
            <span class="score-text">${pred.score}</span>
            <span class="probability">${pred.probability}%</span>
        `;

        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        bar.innerHTML = `<div class="probability-fill" style="width: ${pred.probability}%"></div>`;
        item.appendChild(bar);

        container.appendChild(item);
    });
}

function displayTotalGoals(totals) {
    const container = document.getElementById('total-goals-predictions');
    container.innerHTML = '';

    const items = [
        { label: 'Under 1.5 Goals', value: totals['under_1.5'] },
        { label: 'Under 2.5 Goals', value: totals['under_2.5'] },
        { label: 'Over 2.5 Goals', value: totals['over_2.5'] },
        { label: 'Over 3.5 Goals', value: totals['over_3.5'] }
    ];

    items.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'score-item';
        div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        div.innerHTML = `
            <span class="score-text">${item.label}</span>
            <span class="probability">${item.value}%</span>
        `;

        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        bar.innerHTML = `<div class="probability-fill" style="width: ${item.value}%"></div>`;
        div.appendChild(bar);

        container.appendChild(div);
    });
}

// Match Type Logic
let isCupMatch = false;

function setMatchType(type) {
    isCupMatch = (type === 'cup');

    // Update buttons
    document.getElementById('btn-league').classList.toggle('active', !isCupMatch);
    document.getElementById('btn-cup').classList.toggle('active', isCupMatch);

    // Show/hide tournament context
    const tournamentContext = document.getElementById('tournament-context');
    if (isCupMatch) {
        tournamentContext.style.display = 'block';
    } else {
        tournamentContext.style.display = 'none';
    }

    // Refresh visibility of all inputs based on new state
    updateFifaRankingVisibility();

    // Trigger autosave
    saveState();
}

// Update FIFA ranking section visibility AND Legacy Inputs
function updateFifaRankingVisibility() {
    const compType = document.getElementById('comp-type').value;
    const fifaSection = document.getElementById('fifa-ranking-section');

    // 1. Handle FIFA Section Visibility
    // Only show if Cup AND International
    if (isCupMatch && compType === 'INTERNATIONAL') {
        fifaSection.style.display = 'block';
    } else {
        fifaSection.style.display = 'none';
    }

    // 2. Handle Legacy Rank Input Visibility
    // Hide ONLY if Cup AND International (because we use FIFA inputs then)
    // Show for League and Club Cup
    const shouldHideLegacy = (isCupMatch && compType === 'INTERNATIONAL');

    document.querySelectorAll('.rank-input').forEach(input => {
        input.style.display = shouldHideLegacy ? 'none' : '';

        // Update placeholders/titles based on context
        if (isCupMatch) {
            // Club Cup
            input.placeholder = 'Rank';
            input.max = 210;
            input.title = "Ranking";
        } else {
            // League
            input.placeholder = 'Rank (1-20)';
            input.max = 20;
            input.title = "League Position";
        }
    });

    document.querySelectorAll('.rank-separator').forEach(sep => {
        sep.style.display = shouldHideLegacy ? 'none' : 'inline';
    });
}

// Select Competition Type (Premium UI)
function selectCompType(type) {
    // Update Hidden Input
    document.getElementById('comp-type').value = type;

    // Update Visuals
    document.querySelectorAll('.comp-card').forEach(card => card.classList.remove('active'));
    document.getElementById(`card-${type}`).classList.add('active');

    // Update FIFA Visibility
    updateFifaRankingVisibility();
}

// Select Tournament Stage (Premium UI)
function selectStage(stage) {
    // Update Hidden Input
    document.getElementById('tournament-stage').value = stage;

    // Update Visuals
    document.querySelectorAll('.stage-node').forEach(node => node.classList.remove('active'));
    document.getElementById(`stage-${stage}`).classList.add('active');
}

// Add event listener for competition type change
document.addEventListener('DOMContentLoaded', () => {
    // Initialize correct state for FIFA visibility
    // No need to add change listener to hidden input, selectCompType handles it.
});

let calcMode = 'budget'; // 'budget' or 'profit'

function setCalcMode(mode, btn) {
    calcMode = mode;

    // Update buttons
    const buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(b => b.classList.remove('active'));

    // Handle button activation safely
    if (btn) {
        btn.classList.add('active');
    } else {
        // Fallback for initial state or programmatic calls
        const targetBtn = Array.from(buttons).find(b => b.textContent.includes(mode === 'budget' ? 'Budget' : 'Profit'));
        if (targetBtn) targetBtn.classList.add('active');
    }

    // Update label
    const label = document.getElementById('amount-label');
    if (mode === 'budget') {
        label.textContent = 'Total Budget ($)';
    } else {
        label.textContent = 'Target Profit ($)';
    }
}

function clearCalculator() {
    const container = document.getElementById('odds-rows');
    container.innerHTML = '';
    // Add default empty row
    addOddRow();
}

function loadPreset(type) {
    const container = document.getElementById('odds-rows');
    // REMOVED: container.innerHTML = '';  <-- Now Appends!

    // If container is empty (or just has empty default), maybe clear it first? 
    // Actually, explicit appending is better for "mixing".
    // Let's just remove empty rows if they are untouched before appending? 
    // Simplest is direct append.

    let configs = [];

    switch (type) {
        case '1x2':
            configs = ['Home Win', 'Draw', 'Away Win'];
            break;
        case 'cs_ft': // Full Time Correct Score
            configs = ['1-0', '2-0', '2-1', '0-0', '1-1', '0-1', '0-2', '1-2']; // Top 8 common
            break;
        case 'cs_ht': // Half Time Correct Score
            configs = ['HT 0-0', 'HT 1-0', 'HT 0-1', 'HT 1-1', 'HT 2-0']; // Common HT scores
            break;
        case 'ou':
            configs = ['Over 2.5', 'Under 2.5'];
            break;
        default:
            configs = ['Option 1', 'Option 2'];
    }

    configs.forEach((name, idx) => {
        const div = document.createElement('div');
        div.className = 'odds-row';
        div.style.animation = `fadeInUp 0.3s ease-out ${idx * 0.05}s both`;
        div.innerHTML = `
            <input type="text" placeholder="Option" class="calc-name" value="${name}">
            <input type="number" placeholder="Odds" class="calc-odd" step="0.01">
            <button class="btn-remove" onclick="removeRow(this)">×</button>
        `;
        container.appendChild(div);
    });

    // Focus first odd input
    setTimeout(() => {
        const firstInput = container.querySelector('.calc-odd');
        if (firstInput) firstInput.focus();
    }, 100);
}

function addOddRow() {
    const container = document.getElementById('odds-rows');
    const div = document.createElement('div');
    div.className = 'odds-row';
    div.innerHTML = `
        <input type="text" placeholder="Option" class="calc-name">
        <input type="number" placeholder="Odds" class="calc-odd" step="0.01">
        <button class="btn-remove" onclick="removeRow(this)">×</button>
    `;
    container.appendChild(div);
    div.querySelector('.calc-name').focus();
}

function displayInsights(insights) {
    const container = document.getElementById('insights');
    container.innerHTML = '';

    const items = [
        { label: 'Match Tempo', value: insights.tempo },
        { label: 'Early Goal Likelihood', value: insights.early_goal_likelihood },
        { label: 'Prediction Confidence', value: insights.confidence }
    ];

    items.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'insight-item';
        div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        div.innerHTML = `
            <span class="insight-label">${item.label}:</span>
            <span class="insight-value ${item.value}">${item.value}</span>
        `;

        container.appendChild(div);
    });
}

function displayBothTeamsScore(probability) {
    const container = document.getElementById('both-teams-score');
    container.innerHTML = '';

    const item = document.createElement('div');
    item.className = 'score-item';
    item.style.animation = 'fadeInUp 0.4s ease-out';

    item.innerHTML = `
        <span class="score-text">Probability</span>
        <span class="probability">${probability}%</span>
    `;

    const bar = document.createElement('div');
    bar.className = 'probability-bar';
    bar.innerHTML = `<div class="probability-fill" style="width: ${probability}%"></div>`;
    item.appendChild(bar);

    container.appendChild(item);
}

function displayCleanSheets(cleanSheets, team1, team2) {
    const container = document.getElementById('clean-sheets');
    container.innerHTML = '';

    const items = [
        { label: team1, value: cleanSheets.home },
        { label: team2, value: cleanSheets.away }
    ];

    items.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'score-item';
        div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        div.innerHTML = `
            <span class="score-text">${item.label}</span>
            <span class="probability">${item.value}%</span>
        `;

        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        bar.innerHTML = `<div class="probability-fill" style="width: ${item.value}%"></div>`;
        div.appendChild(bar);

        container.appendChild(div);
    });
}

function displayConfidence(score, breakdown) {
    // Update confidence value
    const valueElement = document.getElementById('confidence-value');
    valueElement.textContent = score ? `${score}/100` : '-';

    // Update gauge fill
    const gaugeFill = document.getElementById('confidence-gauge-fill');
    const percentage = score || 0;
    gaugeFill.style.width = `${percentage}%`;

    // Color based on score
    let color;
    if (percentage >= 75) {
        color = '#22c55e'; // Green
    } else if (percentage >= 50) {
        color = '#f59e0b'; // Orange
    } else {
        color = '#ef4444'; // Red
    }
    gaugeFill.style.background = `linear-gradient(90deg, ${color}, ${color}dd)`;

    // Display breakdown if available
    const breakdownContainer = document.getElementById('confidence-breakdown');
    breakdownContainer.innerHTML = '';

    if (breakdown && breakdown.home) {
        const items = [
            { label: 'Consistency', value: breakdown.home.consistency },
            { label: 'Sample Size', value: breakdown.home.sample_size },
            { label: 'Trend Stability', value: breakdown.home.trend_stability },
            { label: 'Model Fit', value: breakdown.home.model_fit }
        ];

        items.forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'confidence-item';
            div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

            div.innerHTML = `
                <span class="confidence-label">${item.label}:</span>
                <span class="confidence-number">${item.value?.toFixed(1) || '-'}</span>
            `;
            breakdownContainer.appendChild(div);
        });
    }
}

function displayPredictionIntervals(intervals, team1, team2) {
    const container = document.getElementById('prediction-intervals');
    container.innerHTML = '';

    if (!intervals || (!intervals.home_goals_90 && !intervals.away_goals_90)) {
        container.innerHTML = '<p class="no-data">Advanced statistics not available</p>';
        return;
    }

    const items = [
        {
            label: team1,
            range: intervals.home_goals_90 || [0, 0]
        },
        {
            label: team2,
            range: intervals.away_goals_90 || [0, 0]
        }
    ];

    items.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'interval-item';
        div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        div.innerHTML = `
            <span class="interval-label">${item.label}:</span>
            <span class="interval-range">${item.range[0]} - ${item.range[1]} goals</span>
        `;
        container.appendChild(div);
    });
}

function displayTeamMomentum(insights, team1, team2) {
    const container = document.getElementById('team-momentum');
    container.innerHTML = '';

    if (!insights || (!insights.home_momentum && !insights.away_momentum)) {
        container.innerHTML = '<p class="no-data">Advanced statistics not available</p>';
        return;
    }

    const getMomentumIcon = (momentum) => {
        if (momentum === 'IMPROVING') return '↗️';
        if (momentum === 'DECLINING') return '↘️';
        return '→';
    };

    const items = [
        {
            label: team1,
            momentum: insights.home_momentum || 'STABLE',
            icon: getMomentumIcon(insights.home_momentum)
        },
        {
            label: team2,
            momentum: insights.away_momentum || 'STABLE',
            icon: getMomentumIcon(insights.away_momentum)
        },
        {
            label: 'Match Predictability',
            momentum: insights.match_predictability || 'MEDIUM',
            icon: '🎯'
        }
    ];

    items.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'momentum-item';
        div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        div.innerHTML = `
            <span class="momentum-label">${item.label}:</span>
            <span class="momentum-value ${item.momentum}">${item.icon} ${item.momentum}</span>
        `;
        container.appendChild(div);
    });
}

function displayModelQuality(quality) {
    const container = document.getElementById('model-quality');
    container.innerHTML = '';

    if (!quality || !quality.poisson_fit_home) {
        container.innerHTML = '<p class="no-data">Advanced statistics not available</p>';
        return;
    }

    const items = [
        {
            label: 'Home Team Model Fit',
            value: `${(quality.poisson_fit_home * 100).toFixed(1)}%`,
            raw: quality.poisson_fit_home
        },
        {
            label: 'Away Team Model Fit',
            value: `${(quality.poisson_fit_away * 100).toFixed(1)}%`,
            raw: quality.poisson_fit_away
        },
        {
            label: 'Overall Reliability',
            value: quality.overall_reliability || 'MEDIUM'
        }
    ];

    items.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'quality-item';
        div.style.animation = `fadeInUp 0.4s ease-out ${idx * 0.1}s both`;

        let valueClass = '';
        if (item.raw !== undefined) {
            if (item.raw > 0.4) valueClass = 'HIGH';
            else if (item.raw > 0.2) valueClass = 'MEDIUM';
            else valueClass = 'LOW';
        } else {
            valueClass = item.value;
        }

        div.innerHTML = `
            <span class="quality-label">${item.label}:</span>
            <span class="quality-value ${valueClass}">${item.value}</span>
        `;
        container.appendChild(div);
    });
}

function addFeedback(message, isError = false) {
    // Create temporary feedback element
    const feedback = document.createElement('div');
    feedback.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${isError ? 'var(--error)' : 'var(--success)'};
        color: white;
        border-radius: 10px;
        box-shadow: var(--shadow-lg);
        z-index: 1000;
        animation: fadeInDown 0.3s ease-out;
        font-weight: 600;
    `;
    feedback.textContent = message;

    document.body.appendChild(feedback);

    setTimeout(() => {
        feedback.style.animation = 'fadeInUp 0.3s ease-out reverse';
        setTimeout(() => feedback.remove(), 300);
    }, 3000);
}

// Enter key support
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('input').forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                getPrediction();
            }
        });
    });
});

// ---------------------------------------------------------
// MISSING CALCULATOR FUNCTIONS
// ---------------------------------------------------------

function removeRow(btn) {
    const rows = document.querySelectorAll('.odds-row');
    if (rows.length > 2) {
        btn.parentElement.remove();
    } else {
        addFeedback('You need at least 2 options', true);
    }
}

function calculateStakes() {
    const amount = parseFloat(document.getElementById('calc-amount').value);
    const rows = document.querySelectorAll('.odds-row');
    const resultsContainer = document.getElementById('calc-results');

    if (!amount || amount <= 0) {
        addFeedback('Please enter a valid amount', true);
        return;
    }

    let oddsData = [];
    let valid = true;

    rows.forEach((row, idx) => {
        const name = row.querySelector('.calc-name').value || `Option ${idx + 1}`;
        const odd = parseFloat(row.querySelector('.calc-odd').value);

        if (!odd || odd <= 1) {
            valid = false;
        }

        oddsData.push({ name, odd });
    });

    if (!valid) {
        addFeedback('Please enter valid odds (> 1.0)', true);
        return;
    }

    // Calculation Logic
    let sumIP = 0;
    oddsData.forEach(item => {
        item.ip = 1 / item.odd;
        sumIP += item.ip;
    });

    let totalStake = 0;
    let profit = 0;
    let isGuaranteed = false;
    let headerHTML = '';

    if (calcMode === 'budget') {
        // Dutching Logic
        const returnAmount = amount / sumIP;
        totalStake = amount;
        profit = returnAmount - totalStake;

        oddsData.forEach(item => {
            item.stake = returnAmount / item.odd;
            item.return = returnAmount;
            item.profit = profit;
        });

        if (profit >= 0) {
            headerHTML = `<div class="result-summary success">✅ Guaranteed Profit: $${profit.toFixed(2)} (ROI: ${(profit / totalStake * 100).toFixed(1)}%)</div>`;
        } else {
            // "Guaranteed Loss" sounds like a bug to users. Let's say "Optimized Stakes"
            headerHTML = `<div class="result-summary warning">⚠️ Optimized Stakes (Estimated Return: $${returnAmount.toFixed(2)})<br><small>Note: Odds sum to >100%, so a guaranteed profit is mathematically impossible. These stakes minimize your risk.</small></div>`;
        }

    } else {
        // Target Profit Logic
        if (sumIP >= 1) {
            headerHTML = `<div class="result-summary error">❌ Impossible to guarantee target profit. Odds are too low (Sum Prob: ${(sumIP * 100).toFixed(1)}%).</div>`;
            resultsContainer.innerHTML = headerHTML + `<p style="text-align:center">Try increasing the odds or switching to 'Maximize Budget' to see best loss mitigation.</p>`;
            resultsContainer.classList.remove('hidden');
            return;
        }

        const totalReturn = amount / (1 - sumIP);
        totalStake = totalReturn * sumIP; // Cost

        oddsData.forEach(item => {
            item.stake = totalReturn / item.odd;
            item.return = totalReturn;
            item.profit = amount; // Target profit
        });

        headerHTML = `<div class="result-summary success">✅ Required Budget: $${totalStake.toFixed(2)} to win $${amount.toFixed(2)}</div>`;
    }

    // Render Table
    let tableHTML = `
        <table class="calc-table">
            <thead>
                <tr>
                    <th>Option</th>
                    <th>Odds</th>
                    <th>Stake</th>
                    <th>Return</th>
                    <th>Profit</th>
                </tr>
            </thead>
            <tbody>
    `;

    oddsData.forEach(item => {
        tableHTML += `
            <tr>
                <td>${item.name}</td>
                <td>${item.odd.toFixed(2)}</td>
                <td class="stake-val">$${item.stake.toFixed(2)}</td>
                <td>$${item.return.toFixed(2)}</td>
                <td class="profit-val ${item.profit >= 0 ? '' : 'loss'}">$${item.profit.toFixed(2)}</td>
            </tr>
        `;
    });

    tableHTML += `</tbody></table>`;

    resultsContainer.innerHTML = headerHTML + tableHTML;
    resultsContainer.classList.remove('hidden');

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------------------------------------------------
// DATA PERSISTENCE (AUTOSAVE)
// ---------------------------------------------------------

const STORAGE_KEY = 'football_app_state_v1';

function saveState() {
    const state = {
        // Main Form Data
        homeName: document.getElementById('home-name').value,
        awayName: document.getElementById('away-name').value,
        homeTotal: document.getElementById('home-total').value,
        awayTotal: document.getElementById('away-total').value,
        neutralVenue: document.getElementById('neutral-venue').checked,

        homeScored: getInputArrayRaw('home-scored'),
        homeConceded: getInputArrayRaw('home-conceded'),
        homeHalf: getInputArrayRaw('home-first-half'),

        awayScored: getInputArrayRaw('away-scored'),
        awayConceded: getInputArrayRaw('away-conceded'),
        awayHalf: getInputArrayRaw('away-first-half'),

        // Calculator Data
        calcMode: calcMode,
        calcAmount: document.getElementById('calc-amount').value,
        calcRows: []
    };

    // Save Calculator Rows
    document.querySelectorAll('.odds-row').forEach(row => {
        state.calcRows.push({
            name: row.querySelector('.calc-name').value,
            odd: row.querySelector('.calc-odd').value
        });
    });

    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function restoreState() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return;

    try {
        const state = JSON.parse(saved);

        // Restore Form
        if (state.homeName) document.getElementById('home-name').value = state.homeName;
        if (state.homePos) document.getElementById('home-pos').value = state.homePos;
        if (state.homeTotal) document.getElementById('home-total').value = state.homeTotal;
        if (state.awayName) document.getElementById('away-name').value = state.awayName;
        if (state.awayPos) document.getElementById('away-pos').value = state.awayPos;
        if (state.awayTotal) document.getElementById('away-total').value = state.awayTotal;
        if (state.neutralVenue !== undefined) document.getElementById('neutral-venue').checked = state.neutralVenue;
        if (state.isCupMatch !== undefined) setMatchType(state.isCupMatch ? 'cup' : 'league');

        restoreInputArray('home-scored', state.homeScored);
        restoreInputArray('home-conceded', state.homeConceded);
        restoreInputArray('home-first-half', state.homeHalf);

        restoreInputArray('away-scored', state.awayScored);
        restoreInputArray('away-conceded', state.awayConceded);
        restoreInputArray('away-first-half', state.awayHalf);

        // Restore Calculator
        if (state.calcMode) setCalcMode(state.calcMode);
        if (state.calcAmount) document.getElementById('calc-amount').value = state.calcAmount;

        if (state.calcRows && state.calcRows.length > 0) {
            const container = document.getElementById('odds-rows');
            container.innerHTML = ''; // Clear default

            state.calcRows.forEach((row, idx) => {
                const div = document.createElement('div');
                div.className = 'odds-row';
                div.innerHTML = `
                    <input type="text" placeholder="Option" class="calc-name" value="${row.name}">
                    <input type="number" placeholder="Odds" class="calc-odd" step="0.01" value="${row.odd}">
                    <button class="btn-remove" onclick="removeRow(this)">×</button>
                `;
                container.appendChild(div);
            });
        }

        // Setup Auto-Save Listeners on restored elements
        setupAutosave();

        console.log('State restored from localStorage');

    } catch (e) {
        console.error('Error restoring state:', e);
    }
}

// Helpers
function getInputArrayRaw(prefix) {
    const values = [];
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById(`${prefix}-${i}`);
        values.push(el ? el.value : '');
    }
    return values;
}

function restoreInputArray(prefix, values) {
    if (!values) return;
    values.forEach((val, idx) => {
        const el = document.getElementById(`${prefix}-${idx + 1}`);
        if (el) el.value = val;
    });
}

function setupAutosave() {
    // Add change listeners to EVERYTHING
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.oninput = debounce(saveState, 500);
        input.onchange = saveState; // For checkboxes
    });

    // Observer for dynamic calculator rows
    const calcContainer = document.getElementById('odds-rows');
    if (calcContainer) {
        const observer = new MutationObserver(() => {
            saveState();
            // Re-attach listeners to new inputs
            const newInputs = calcContainer.querySelectorAll('input');
            newInputs.forEach(input => input.oninput = debounce(saveState, 500));
        });
        observer.observe(calcContainer, { childList: true, subtree: true });
    }
}

// Debounce util
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ---------------------------------------------------------
// HISTORY MODAL LOGIC
// ---------------------------------------------------------

function toggleHistoryModal() {
    const modal = document.getElementById('history-modal');
    if (modal.style.display === 'block') {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Enable scroll
    } else {
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Disable scroll
        loadHistory();
    }
}

// Close modal when clicking outside
window.onclick = function (event) {
    const modal = document.getElementById('history-modal');
    if (event.target == modal) {
        toggleHistoryModal();
    }
}

async function loadHistory() {
    const tbody = document.getElementById('history-list');
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center">Loading history...</td></tr>';

    try {
        const response = await fetch('/api/history');
        const history = await response.json();
        renderHistoryTable(history);
    } catch (error) {
        console.error('Error loading history:', error);
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: var(--error)">Error loading history</td></tr>';
    }
}

function renderHistoryTable(history) {
    const tbody = document.getElementById('history-list');
    tbody.innerHTML = '';

    if (!history || history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center">No prediction history found.</td></tr>';
        return;
    }

    history.forEach(entry => {
        const row = document.createElement('tr');

        // Determine result badge
        const pred = entry.prediction;
        let resultBadge = '';
        if (pred.home_win > 50) resultBadge = `<span class="history-result-badge badge-win">Home ${pred.home_win}%</span>`;
        else if (pred.away_win > 50) resultBadge = `<span class="history-result-badge badge-loss">Away ${pred.away_win}%</span>`;
        else resultBadge = `<span class="history-result-badge badge-draw">Draw ${pred.draw}%</span>`;

        row.innerHTML = `
            <td>${entry.timestamp || '-'}</td>
            <td>
                <div style="font-weight:600">${entry.home_team}</div>
                <div style="font-size:0.85rem; color:var(--text-secondary)">vs ${entry.away_team}</div>
            </td>
            <td>
                <div style="font-size:0.85rem">🏠 ${entry.home_rank}</div>
                <div style="font-size:0.85rem">✈️ ${entry.away_rank}</div>
            </td>
            <td>
                ${resultBadge}
                <div style="font-size:0.8em; margin-top:2px; opacity:0.8">Conf: ${pred.confidence}</div>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    restoreState();
    setupAutosave();
});
