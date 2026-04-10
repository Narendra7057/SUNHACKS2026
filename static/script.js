const analyzeBtn = document.getElementById('analyzeBtn');
const domainSelect = document.getElementById('domain');
const clauseInput = document.getElementById('clause');
const contractFileInput = document.getElementById('contractFile');
const loadingEl = document.getElementById('loading');
const resultEl = document.getElementById('result');
const govDecisionCard = document.getElementById('govDecisionCard');
const governanceActionEl = document.getElementById('governanceAction');
const riskBadgeEl = document.getElementById('riskBadge');

const originalDecisionEl = document.getElementById('originalDecision');
const correctedDecisionEl = document.getElementById('correctedDecision');
const explanationEl = document.getElementById('explanation');
const trustScoreEl = document.getElementById('trustScore');
const impactSeverityEl = document.getElementById('impactSeverity');
const impactBusinessRiskEl = document.getElementById('impactBusinessRisk');
const impactEstimatedLossEl = document.getElementById('impactEstimatedLoss');
const complianceStatusEl = document.getElementById('complianceStatus');
const complianceReasonEl = document.getElementById('complianceReason');
const auditTimelineEl = document.getElementById('auditTimeline');
const auditLogEl = document.getElementById('auditLog');
const traceCrewAIEl = document.getElementById('traceCrewAI');
const traceOllamaEl = document.getElementById('traceOllama');
const traceLangSmithEl = document.getElementById('traceLangSmith');
const traceOpenTelemetryEl = document.getElementById('traceOpenTelemetry');
const traceIsolationForestEl = document.getElementById('traceIsolationForest');

const DOMAIN_INFO = {
  legal: '⚖️ Legal - Contract Analysis',
  ecommerce: '🛍️ E-commerce - Product Review',
  fintech: '💰 Fintech - Loan Approval',
  healthcare: '🏥 Healthcare - Diagnosis'
};

function safeText(value, fallback = '-') {
  if (value === null || value === undefined) {
    return fallback;
  }
  const text = String(value).trim();
  return text || fallback;
}

function buildHumanExplanation(data) {
  const domain = safeText(data.domain, 'unknown');
  const domainInfo = DOMAIN_INFO[domain] || domain;
  const originalDecision = safeText(data.original_output?.decision, 'UNKNOWN');
  const correctedDecision = safeText(data.corrected_output?.decision, 'UNKNOWN');
  const issue = data.issues_detected?.[0];
  const whatWasWrong = issue?.what_is_wrong || 'No major decision error was found, but governance refinement was still applied.';
  const detectionBasis = `Semantic risk=${safeText(data.risk_level)}; anomaly=${data.anomaly_detected ? 'detected' : 'not detected'}; governance action=${safeText(data.governance_action)}.`;

  return [
    `${domainInfo} Agent first reviewed the input and produced ${originalDecision}.`,
    `The supervisor found a governance concern: ${whatWasWrong}`,
    `Detection signals were combined in real time (${detectionBasis}) to determine if correction was needed.`,
    `TrustGuard then triggered self-reprompting and the domain agent produced the governed result: ${correctedDecision}.`,
  ];
}

function buildAuditSteps(data) {
  const originalDecision = safeText(data.original_output?.decision, 'UNKNOWN');
  const correctedDecision = safeText(data.corrected_output?.decision, 'UNKNOWN');
  const mismatch = data.error_detected || (data.comparison && data.comparison.changed);

  return [
    {
      icon: mismatch ? '' : '',
      title: 'ContractAnalysisAgent',
      status: mismatch ? 'FAILED' : 'SUCCESS',
      details: `Output: ${originalDecision} | Issue: ${mismatch ? 'underestimated risk' : 'initial decision acceptable'}`,
    },
    {
      icon: '',
      title: ' Supervisor Agent (TrustGuard)',
      status: 'DETECTED',
      details: `Semantic risk=${safeText(data.risk_level)} + anomaly=${data.anomaly_detected ? 'detected' : 'not detected'}`,
    },
    {
      icon: '',
      title: 'Self-Reprompt',
      status: 'TRIGGERED',
      details: 'Re-evaluated with strict compliance rules and audit standardization.',
    },
    {
      icon: '',
      title: 'DomainAgent',
      status: 'SUCCESS',
      details: `Corrected to ${correctedDecision}`,
    },
  ];
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result || '').toString());
    reader.onerror = () => reject(new Error('Could not read uploaded file.'));
    reader.readAsText(file);
  });
}

async function getAnalysisInput() {
  const textValue = clauseInput.value.trim();
  const file = contractFileInput.files && contractFileInput.files[0];

  if (file) {
    const fileText = (await readFileAsText(file)).trim();
    if (fileText) {
      return fileText;
    }
  }

  return textValue;
}

async function analyzeInput() {
  const input = await getAnalysisInput();
  if (!input) {
    alert('Please paste input or upload a .txt file.');
    return;
  }

  loadingEl.classList.remove('hidden');
  resultEl.classList.add('hidden');
  resultEl.classList.remove('fade-in');

  try {
    const domain = domainSelect.value || 'legal';
    const response = await fetch('/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input, domain }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Request failed');
    }

    const originalReason = safeText(data.original_output?.reasoning, 'No reasoning provided.');
    const correctedReason = safeText(data.corrected_output?.reasoning, 'No reasoning provided.');
    const governanceAction = (data.governance_action || 'ALLOW').toUpperCase();
    const riskLevel = safeText(data.risk_level);

    governanceActionEl.textContent = governanceAction;
    riskBadgeEl.textContent = `Risk: ${riskLevel}`;

    governanceActionEl.classList.remove('block', 'allow', 'review');
    govDecisionCard.classList.remove('block', 'allow', 'review');
    if (governanceAction === 'BLOCK') {
      governanceActionEl.classList.add('block');
      govDecisionCard.classList.add('block');
    } else if (governanceAction === 'REVIEW') {
      governanceActionEl.classList.add('review');
      govDecisionCard.classList.add('review');
    } else {
      governanceActionEl.classList.add('allow');
      govDecisionCard.classList.add('allow');
    }

    originalDecisionEl.textContent = `${safeText(data.original_output?.decision)} — ${originalReason}`;
    correctedDecisionEl.textContent = `${safeText(data.corrected_output?.decision)} — ${correctedReason}`;
    const explanationParagraphs = buildHumanExplanation(data);
    explanationEl.innerHTML = explanationParagraphs.map((paragraph) => `<p>${paragraph}</p>`).join('');
    trustScoreEl.textContent = `${data.trust_score}`;

    impactSeverityEl.textContent = data.impact?.severity || '-';
    impactBusinessRiskEl.textContent = data.impact?.business_risk || '-';
    impactEstimatedLossEl.textContent = data.impact?.estimated_loss || '-';

    complianceStatusEl.textContent = data.compliance?.status || '-';
    complianceReasonEl.textContent = data.compliance?.reason || '-';

    const trace = data.system_trace || {};
    traceCrewAIEl.textContent = safeText(trace.CrewAI);
    traceOllamaEl.textContent = safeText(trace.Ollama);
    traceLangSmithEl.textContent = safeText(trace.LangSmith);
    traceOpenTelemetryEl.textContent = safeText(trace.OpenTelemetry);
    traceIsolationForestEl.textContent = safeText(trace.IsolationForest);

    auditTimelineEl.innerHTML = '';
    const auditSteps = buildAuditSteps(data);
    auditSteps.forEach((item) => {
      const li = document.createElement('li');
      li.className = `timeline-card status-${item.status.toLowerCase()}`;
      li.innerHTML = `
        <div class="timeline-header">
          <span class="timeline-icon">${item.icon}</span>
          <span class="step">${item.title}</span>
          <span class="status-chip">${item.status}</span>
        </div>
        <div class="action">${item.details}</div>
      `;
      auditTimelineEl.appendChild(li);
    });

    auditLogEl.textContent = '';

    resultEl.classList.remove('hidden');
    resultEl.classList.add('fade-in');
  } catch (error) {
    alert(`Error: ${error.message}`);
  } finally {
    loadingEl.classList.add('hidden');
  }
}

analyzeBtn.addEventListener('click', analyzeInput);
