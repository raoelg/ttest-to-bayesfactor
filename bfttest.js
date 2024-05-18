// Utility functions for log operations
function logExpXminusExpY(logX, logY) {
  return logX > logY
    ? logX + Math.log1p(-Math.exp(logY - logX))
    : logY + Math.log1p(-Math.exp(logX - logY));
}

function logExpXplusExpY(logX, logY) {
  return logX > logY
    ? logX + Math.log1p(Math.exp(logY - logX))
    : logY + Math.log1p(Math.exp(logX - logY));
}

function sumWithPropErr(x1, x2, err1, err2) {
  const logAbs1 = x1 + Math.log(err1);
  const logAbs2 = x2 + Math.log(err2);
  const logSum = logExpXplusExpY(x1, x2);
  const absSum = 0.5 * logExpXplusExpY(2 * logAbs1, 2 * logAbs2);
  const propErr = Math.exp(absSum - logSum);
  return [logSum, propErr];
}

// Prior values function
function rpriorValues(modelType, priorType) {
  const priorValues = {
    ttestTwo: {
      ultrawide: Math.sqrt(2),
      wide: 1,
      medium: Math.sqrt(2) / 2,
    },
    ttestOne: {
      ultrawide: Math.sqrt(2),
      wide: 1,
      medium: Math.sqrt(2) / 2,
    },
    // Add other model types if needed
  };
  return priorValues[modelType]?.[priorType] ?? null;
}

// Meta-analysis functions
function meta_bf_interval(lower, upper, t, N, df, rscale) {
  const nullLike = jStat.studentt.pdf(t, df, 0, true);
  const logPriorProbs = [lower, upper].map((bound) =>
    Math.log(jStat.cauchy.cdf(bound, 0, rscale)),
  );
  const priorInterval = logExpXminusExpY(logPriorProbs[1], logPriorProbs[0]);
  const deltaEst = t / Math.sqrt(N);
  const meanDelta = deltaEst;
  const logConst = meta_t_like(meanDelta, t, N, df, rscale, true);
  const integral = math.integrate(
    (delta) => meta_t_like(delta, t, N, df, rscale, true),
    lower - meanDelta,
    upper - meanDelta,
  );
  const val = Math.log(integral.result) + logConst - priorInterval - nullLike;
  const err = Math.exp(Math.log(integral.error) - val);
  return { bf: val, properror: err, method: "quadrature" };
}

function meta_bf_interval_approx(lower, upper, t, N, df, rscale) {
  const deltaEst = t / Math.sqrt(N);
  const meanDelta = deltaEst;
  const varDelta = 1 / N;
  const logPriorProbs = [lower, upper].map((bound) =>
    Math.log(jStat.cauchy.cdf(bound, 0, rscale)),
  );
  const logPostProbs = [lower, upper].map((bound) =>
    Math.log(jStat.studentt.cdf((bound - meanDelta) / Math.sqrt(varDelta), df)),
  );
  const priorInterval = logExpXminusExpY(logPriorProbs[1], logPriorProbs[0]);
  const postInterval = logExpXminusExpY(logPostProbs[1], logPostProbs[0]);
  const logBfInterval = postInterval - priorInterval;
  const logBfPoint = meta_bf_interval(-Infinity, Infinity, t, N, df, rscale);
  return lower === -Infinity && upper === Infinity
    ? {
        bf: logBfPoint.bf,
        properror: logBfPoint.properror,
        method: "Savage-Dickey t approximation",
      }
    : {
        bf: logBfInterval + logBfPoint.bf,
        properror: NaN,
        method: "Savage-Dickey t approximation",
      };
}

function meta_t_bf(t, N, df, options) {
  const { interval = null, rscale, complement = false } = options;

  if (interval && interval.length !== 2) {
    throw new Error("argument interval must have two elements.");
  }

  const metaBfIntervalFn =
    Math.abs(t) > 15 ? meta_bf_interval_approx : meta_bf_interval;

  if (!interval) {
    return metaBfIntervalFn(-Infinity, Infinity, t, N, df, rscale);
  }

  const [lower, upper] = interval.sort((a, b) => a - b);

  if (lower === -Infinity && upper === Infinity) {
    return complement
      ? { bf: NaN, properror: NaN, method: NaN }
      : metaBfIntervalFn(-Infinity, Infinity, t, N, df, rscale);
  }

  if (Math.abs(t) > 5) {
    return meta_bf_interval_approx(lower, upper, t, N, df, rscale);
  }

  let bf, bfCompl;
  if (isFinite(lower) && isFinite(upper)) {
    const logPriorProbs = [-Infinity, lower, upper, Infinity].map((bound) =>
      Math.log(jStat.cauchy.cdf(bound, 0, rscale)),
    );
    const priorInterval1 = logExpXminusExpY(logPriorProbs[1], logPriorProbs[0]);
    const priorInterval3 = logExpXminusExpY(logPriorProbs[3], logPriorProbs[2]);
    const priorInterval13 = logExpXplusExpY(priorInterval1, priorInterval3);
    const bf1 = metaBfIntervalFn(-Infinity, lower, t, N, df, rscale);
    const bf2 = metaBfIntervalFn(lower, upper, t, N, df, rscale);
    const bf3 = metaBfIntervalFn(upper, Infinity, t, N, df, rscale);

    if (complement) {
      bfCompl = sumWithPropErr(
        bf1.bf + priorInterval1,
        bf3.bf + priorInterval3,
        bf1.properror,
        bf3.properror,
      );
      bfCompl[0] -= priorInterval13;
    }

    bf = bf2;
  } else {
    bf = complement
      ? metaBfIntervalFn(lower, upper, t, N, df, rscale)
      : metaBfIntervalFn(-Infinity, upper, t, N, df, rscale);
  }

  return complement
    ? {
        bf: bfCompl[0],
        properror: bfCompl[1],
        method: "Savage-Dickey t approximation",
      }
    : { bf: bf.bf, properror: bf.properror, method: bf.method };
}

function meta_t_like(delta, t, N, df, rscale, log = false) {
  const density = jStat.studentt.pdf(t, df, delta / rscale);
  return log ? Math.log(density) : density;
}

// Main t-test function
function ttest_tstat(
  t,
  n1,
  n2 = 0,
  nullInterval = null,
  rscale = "medium",
  complement = false,
  simple = false,
) {
  const rscaleValue = rpriorValues(n2 ? "ttestTwo" : "ttestOne", rscale);
  if (!rscaleValue) throw new Error("Unknown prior type.");

  if (t.length !== 1 || n1.length !== 1)
    throw new Error("Invalid input lengths");

  const nu = n2 === 0 ? n1 - 1 : n1 + n2 - 2;
  const n =
    n2 === 0 ? n1 : Math.exp(Math.log(n1) + Math.log(n2) - Math.log(n1 + n2));

  if (n < 1 || nu < 1) throw new Error("not enough observations");
  if (!isFinite(t)) throw new Error("data are essentially constant");

  let res;
  try {
    res = meta_t_bf(t, n, nu, {
      interval: nullInterval,
      rscale: rscaleValue,
      complement: complement,
    });
  } catch (error) {
    console.error(error);
    return { bf: NaN, properror: NaN, method: NaN };
  }

  return simple ? { B10: Math.exp(res.bf) } : res;
}

function calculateTTest() {
  const t = parseFloat(document.getElementById("t").value);
  const n1 = parseFloat(document.getElementById("n1").value);
  const n2 = parseFloat(document.getElementById("n2").value) || 0;
  const nullInterval = document
    .getElementById("nullInterval")
    .value.split(",")
    .map(parseFloat);
  const rscale = document.getElementById("rscale").value;
  const complement = document.getElementById("complement").checked;
  const simple = document.getElementById("simple").checked;

  const result = ttest_tstat(
    [t],
    [n1],
    n2,
    nullInterval,
    rscale,
    complement,
    simple,
  );
  document.getElementById("result").textContent = JSON.stringify(
    result,
    null,
    2,
  );
  return false;
}
