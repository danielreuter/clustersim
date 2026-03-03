// ---------------------------------------------------------------------------
// Transformer Model definitions – ported from erdil.py (Erdil 2025)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/**
 * Estimates the read/write bytes for a single matmul tile under TP.
 *
 * Scalar version of the numpy‑based Python original.
 */
export function matmulRwBytes(
  m: number,
  k: number,
  n: number,
  wpBytes: number = 1,
  apBytes: number = 1,
  tp: number = 1,
): number {
  const tp1 = Math.min(Math.max(Math.sqrt((m * tp) / k), 1), tp);
  const tp2 = tp / tp1;
  return m * k * wpBytes + tp1 * k * n * apBytes + tp2 * m * n * apBytes;
}

// ---------------------------------------------------------------------------
// Model class
// ---------------------------------------------------------------------------

export class Model {
  // Direct fields
  name: string;
  dModel: number;
  dFf: number;
  layers: number;
  nExperts: number;
  nActiveExperts: number;
  numQueryHeads: number;
  groupSize: number;
  mla: boolean;
  mlaKvCompressionDim: number | null;
  mlaQueryCompressionDim: number | null;
  weightPrecisionBytes: number;
  activationPrecisionBytes: number;
  vocabSize: number;
  ffMatrixCount: [number, number];
  parallelAttention: boolean;

  // Derived fields
  ffParamsPerLayerPerExpert: number;
  sparsityFactor: number;
  totalFfParams: number;
  numKvHeads: number;
  dHead: number;
  dAllAttnHeads: number;
  attnParamsPerLayer: number;
  embeddingParams: number;
  totalAttnParams: number;
  totalParams: number;
  totalActiveParams: number;
  kvCacheSizePerInputBytes: number;

  constructor(opts: {
    name?: string;
    dModel?: number;
    dFf?: number;
    ffMatrixCount?: [number, number];
    layers?: number;
    nExperts?: number;
    nActiveExperts?: number;
    numQueryHeads?: number;
    groupSize?: number;
    mla?: boolean;
    mlaKvCompressionDim?: number | null;
    mlaQueryCompressionDim?: number | null;
    weightPrecisionBytes?: number;
    activationPrecisionBytes?: number;
    dHead?: number | null;
    vocabSize?: number;
    parallelAttention?: boolean;
  }) {
    this.name = opts.name ?? "None";
    this.dModel = opts.dModel ?? 3 * 2 ** 12;
    this.dFf = opts.dFf ?? 9 * 2 ** 12;
    this.ffMatrixCount = opts.ffMatrixCount ?? [1, 1];
    this.layers = opts.layers ?? 120;
    this.nExperts = opts.nExperts ?? 1;
    this.nActiveExperts = opts.nActiveExperts ?? 1;
    this.numQueryHeads = opts.numQueryHeads ?? 128;
    this.groupSize = opts.groupSize ?? 1;
    this.mla = opts.mla ?? false;
    this.mlaKvCompressionDim = opts.mlaKvCompressionDim ?? null;
    this.mlaQueryCompressionDim = opts.mlaQueryCompressionDim ?? null;
    this.weightPrecisionBytes = opts.weightPrecisionBytes ?? 2;
    this.activationPrecisionBytes = opts.activationPrecisionBytes ?? 2;
    this.vocabSize = opts.vocabSize ?? 0;
    this.parallelAttention = opts.parallelAttention ?? false;

    if (this.numQueryHeads % this.groupSize !== 0) {
      throw new Error("numQueryHeads must be divisible by groupSize");
    }
    if (this.mla && this.groupSize !== 1) {
      throw new Error("MLA requires groupSize === 1");
    }

    // Derived
    this.ffParamsPerLayerPerExpert =
      (this.ffMatrixCount[0] + this.ffMatrixCount[1]) * this.dModel * this.dFf;
    this.sparsityFactor = Math.floor(this.nExperts / this.nActiveExperts);
    this.totalFfParams = this.layers * this.nExperts * this.ffParamsPerLayerPerExpert;
    this.numKvHeads = (2 * this.numQueryHeads) / this.groupSize;
    this.dHead = opts.dHead != null ? opts.dHead : this.dModel / this.numQueryHeads;
    this.dAllAttnHeads = (this.numQueryHeads + this.numKvHeads) * this.dHead;

    if (this.mla) {
      const kvDim = this.mlaKvCompressionDim!;
      const qDim = this.mlaQueryCompressionDim!;
      this.attnParamsPerLayer =
        2 * kvDim * this.dModel +
        qDim * this.dModel +
        this.numKvHeads * this.dHead * kvDim +
        this.numQueryHeads * this.dHead * qDim +
        this.dHead * this.numQueryHeads * this.dModel;
    } else {
      this.attnParamsPerLayer =
        this.dAllAttnHeads * this.dModel +
        this.dHead * this.numQueryHeads * this.dModel;
    }

    this.embeddingParams = this.vocabSize * this.dModel * 2;
    this.totalAttnParams = this.layers * this.attnParamsPerLayer;
    this.totalParams = this.totalAttnParams + this.totalFfParams + this.embeddingParams;
    this.totalActiveParams =
      this.totalAttnParams +
      this.totalFfParams / this.sparsityFactor +
      this.embeddingParams;

    if (this.mla) {
      this.kvCacheSizePerInputBytes =
        this.mlaKvCompressionDim! * this.layers * this.activationPrecisionBytes;
    } else {
      this.kvCacheSizePerInputBytes =
        this.numKvHeads * this.dHead * this.layers * this.activationPrecisionBytes;
    }
  }

  // -------------------------------------------------------------------
  // Cost helpers
  // -------------------------------------------------------------------

  arithmeticCostFlop(inputLen: number, batchSize: number, seqLen: number = 1): number {
    const meanInputLen = (inputLen + (inputLen + seqLen - 1)) / 2;

    if (this.mla) {
      return (
        2 * this.totalActiveParams * batchSize * seqLen +
        4 * this.mlaKvCompressionDim! * this.numQueryHeads * this.layers * meanInputLen * batchSize * seqLen
      );
    }
    return (
      2 * this.totalActiveParams * batchSize * seqLen +
      4 * this.dHead * this.numQueryHeads * this.layers * meanInputLen * batchSize * seqLen
    );
  }

  ffdFlop(batchSize: number, seqLen: number = 1): number {
    return (2 * this.totalFfParams / this.sparsityFactor) * batchSize * seqLen;
  }

  attnKvFlop(inputLen: number, batchSize: number, seqLen: number = 1): number {
    const meanInputLen = (inputLen + (inputLen + seqLen - 1)) / 2;

    if (this.mla) {
      return (
        4 * this.mlaKvCompressionDim! * this.numQueryHeads * this.layers * meanInputLen * batchSize * seqLen
      );
    }
    return (
      4 * this.dHead * this.numQueryHeads * this.layers * meanInputLen * batchSize * seqLen
    );
  }

  attnProjFlop(inputLen: number, batchSize: number, seqLen: number = 1): number {
    return (
      this.arithmeticCostFlop(inputLen, batchSize, seqLen) -
      (this.ffdFlop(batchSize, seqLen) + this.attnKvFlop(inputLen, batchSize, seqLen))
    );
  }

  memoryReadsWritesBytes(
    inputLen: number,
    batchSize: number,
    seqLen: number,
    tp: number,
  ): number {
    const batchSizeTokens = batchSize * seqLen;
    const kvCacheSizeBytes = this.kvCacheSizePerInputBytes * inputLen * batchSize;

    const usedExpertsFraction = 1 - (1 - 1 / this.sparsityFactor) ** batchSizeTokens;
    const wp = this.weightPrecisionBytes;
    const ap = this.activationPrecisionBytes;

    const feedforwardMatmulRwBytes =
      usedExpertsFraction *
      this.nExperts *
      (this.ffMatrixCount[0] + this.ffMatrixCount[1]) *
      matmulRwBytes(this.dModel, this.dFf, batchSizeTokens / this.sparsityFactor, wp, ap, tp);

    let attentionRwBytes: number;

    if (this.mla) {
      const qkvCompressed = matmulRwBytes(
        this.mlaKvCompressionDim! + this.mlaQueryCompressionDim!,
        this.dModel,
        batchSizeTokens,
        wp,
        ap,
        tp,
      );
      const qkv = matmulRwBytes(
        this.dAllAttnHeads,
        this.mlaKvCompressionDim! + this.mlaQueryCompressionDim!,
        batchSizeTokens,
        wp,
        ap,
        tp,
      );
      const proj = matmulRwBytes(this.dModel, this.dHead * this.numQueryHeads, batchSizeTokens, wp, ap, tp);
      attentionRwBytes = qkvCompressed + qkv + proj;
    } else {
      const qkv = matmulRwBytes(this.dAllAttnHeads, this.dModel, batchSizeTokens, wp, ap, tp);
      const proj = matmulRwBytes(this.dModel, this.dHead * this.numQueryHeads, batchSizeTokens, wp, ap, tp);
      attentionRwBytes = qkv + proj;
    }

    const unembeddingBytes = wp * this.vocabSize * this.dModel;

    return kvCacheSizeBytes + this.layers * (feedforwardMatmulRwBytes + attentionRwBytes) + unembeddingBytes;
  }
}

// ---------------------------------------------------------------------------
// scale_model
// ---------------------------------------------------------------------------

export function scaleModel(
  model: Model,
  scaleFactor: number,
  depthExponent: number = 1 / 3,
): Model {
  const dModel = model.dModel * scaleFactor ** ((1 - depthExponent) / 2);
  const dFf = model.dFf * scaleFactor ** ((1 - depthExponent) / 2);
  const layers = Math.round(model.layers * scaleFactor ** depthExponent);

  const numQueryHeads = Math.ceil(
    model.numQueryHeads * scaleFactor ** ((1 - depthExponent) / 4),
  );
  const numGroups = model.numQueryHeads / model.groupSize;
  const groupSize = numQueryHeads / numGroups;

  return new Model({
    name: model.name,
    dModel,
    dFf,
    ffMatrixCount: model.ffMatrixCount,
    layers,
    nExperts: model.nExperts,
    nActiveExperts: model.nActiveExperts,
    numQueryHeads,
    groupSize,
    dHead: model.dHead * scaleFactor ** ((1 - depthExponent) / 4),
    weightPrecisionBytes: model.weightPrecisionBytes,
    activationPrecisionBytes: model.activationPrecisionBytes,
    vocabSize: model.vocabSize,
    parallelAttention: model.parallelAttention,
  });
}

// ---------------------------------------------------------------------------
// Model definitions
// ---------------------------------------------------------------------------

export const GPT_4 = new Model({
  name: "GPT-4",
  dModel: 12288,
  dFf: 3 * 12288,
  layers: 120,
  nExperts: 16,
  nActiveExperts: 2,
  numQueryHeads: 96,
  groupSize: 96,
  dHead: Math.floor((3 * 12288) / (2 * 96)),
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 100256,
});

export const GPT_3_5 = new Model({
  name: "GPT 3.5",
  dModel: 2 ** 9 * 3 ** 2,
  dFf: 4 * 2 ** 9 * 3 ** 2,
  layers: 32,
  nExperts: 4,
  nActiveExperts: 2,
  numQueryHeads: 32,
  groupSize: 32,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
});

export const GPT_3 = new Model({
  name: "GPT-3",
  dModel: 12288,
  dFf: 4 * 12288,
  layers: 96,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 96,
  dHead: 128,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 50257,
});

export const PaLM_540B = new Model({
  name: "PaLM 540B",
  dModel: 18432,
  dFf: 73728,
  ffMatrixCount: [2, 1],
  layers: 118,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 48,
  dHead: 256,
  groupSize: 48,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 256000,
  parallelAttention: true,
});

export const PaLM_8B = new Model({
  name: "PaLM 8B",
  dModel: 4096,
  dFf: 4 * 4096,
  ffMatrixCount: [2, 1],
  layers: 32,
  numQueryHeads: 16,
  groupSize: 16,
  dHead: 256,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 1,
  vocabSize: 256000,
  parallelAttention: true,
});

export const Falcon = new Model({
  name: "Falcon 180B",
  dModel: 14848,
  dFf: 4 * 14848,
  layers: 80,
  dHead: 64,
  numQueryHeads: 232,
  groupSize: 232,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 65024,
});

export const GPT_J_6B = new Model({
  name: "GPT-J 6B",
  dModel: 4096,
  dFf: 4 * 4096,
  layers: 28,
  numQueryHeads: 16,
  dHead: 256,
  groupSize: 1,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 50257,
});

export const Mixtral_8x22B = new Model({
  name: "Mixtral 8x22B",
  dModel: 6144,
  dFf: 16384,
  ffMatrixCount: [2, 1],
  layers: 56,
  nExperts: 8,
  nActiveExperts: 2,
  numQueryHeads: 48,
  dHead: 128,
  groupSize: 6,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 32000,
});

export const Mixtral_8x22B_MQA = new Model({
  name: "Mixtral 8x22B MQA",
  dModel: 6144,
  dFf: 16384,
  ffMatrixCount: [2, 1],
  layers: 56,
  nExperts: 8,
  nActiveExperts: 2,
  numQueryHeads: 48,
  dHead: 128,
  groupSize: 48,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 32000,
});

export const Mixtral_8x7B = new Model({
  name: "Mixtral 8x7B",
  dModel: 4096,
  dFf: 14336,
  ffMatrixCount: [2, 1],
  layers: 32,
  nExperts: 8,
  nActiveExperts: 2,
  numQueryHeads: 32,
  groupSize: 4,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 32000,
});

export const Mistral_Large_2 = new Model({
  name: "Mistral Large 2",
  dModel: 12288,
  dFf: 28672,
  ffMatrixCount: [2, 1],
  layers: 88,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 96,
  groupSize: 12,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 32768,
});

export const Llama_3_8B = new Model({
  name: "LLaMa 3 8B",
  dModel: 4096,
  dFf: 14336,
  ffMatrixCount: [2, 1],
  layers: 32,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 32,
  groupSize: 4,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 128256,
});

export const Llama_3_8B_MQA = new Model({
  name: "LLaMa 3 8B MQA",
  dModel: 4096,
  dFf: 14336,
  ffMatrixCount: [2, 1],
  layers: 32,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 32,
  groupSize: 32,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 128256,
});

export const Llama_3_8B_8_bit = new Model({
  name: "LLaMa 3 8B",
  dModel: 4096,
  dFf: 14336,
  ffMatrixCount: [2, 1],
  layers: 32,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 32,
  groupSize: 4,
  activationPrecisionBytes: 1,
  weightPrecisionBytes: 2,
  vocabSize: 128256,
});

export const Llama_3_70B = new Model({
  name: "LLaMa 3 70B",
  dModel: 8192,
  dFf: 28672,
  ffMatrixCount: [2, 1],
  layers: 80,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 64,
  groupSize: 8,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 128256,
});

export const Llama_3_70B_MQA = new Model({
  name: "LLaMa 3 70B MQA",
  dModel: 8192,
  dFf: 28672,
  ffMatrixCount: [2, 1],
  layers: 80,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 64,
  groupSize: 64,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 128256,
});

export const Llama_3_70B_8_bit = new Model({
  name: "LLaMa 3 70B 8-bit",
  dModel: 8192,
  dFf: 28672,
  ffMatrixCount: [2, 1],
  layers: 80,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 64,
  groupSize: 8,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 1,
  vocabSize: 128256,
});

export const Llama_3_70B_4_bit = new Model({
  name: "LLaMa 3 70B 4-bit",
  dModel: 8192,
  dFf: 28672,
  ffMatrixCount: [2, 1],
  layers: 80,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 64,
  groupSize: 8,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 0.5,
  vocabSize: 128256,
});

export const Llama_3_405B = new Model({
  name: "LLaMa 3 405B",
  dModel: 16384,
  dFf: 53248,
  ffMatrixCount: [2, 1],
  layers: 126,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 128,
  groupSize: 8,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 2,
  vocabSize: 128256,
});

export const Llama_3_405B_8_bit = new Model({
  name: "LLaMa 3 405B",
  dModel: 16384,
  dFf: 53248,
  ffMatrixCount: [2, 1],
  layers: 126,
  nExperts: 1,
  nActiveExperts: 1,
  numQueryHeads: 128,
  groupSize: 8,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 1,
  vocabSize: 128256,
});

export const DeepSeek_V3 = new Model({
  name: "DeepSeek V3",
  dModel: 7168,
  dFf: 2048,
  ffMatrixCount: [2, 1],
  layers: 58,
  nExperts: 256,
  nActiveExperts: 9,
  dHead: 128,
  numQueryHeads: 128,
  mla: true,
  mlaKvCompressionDim: 512,
  mlaQueryCompressionDim: 1536,
  activationPrecisionBytes: 2,
  weightPrecisionBytes: 1,
  vocabSize: 129280,
});

// ---------------------------------------------------------------------------
// Scaled variants
// ---------------------------------------------------------------------------

export const GPT_4_100T_param = (() => {
  const m = scaleModel(GPT_4, 50);
  m.name = "GPT 4 with 100T params";
  return m;
})();

export const GPT_4_2x = scaleModel(GPT_4, 1 / 2);
export const GPT_4_4x = scaleModel(GPT_4, 1 / 4);
export const GPT_4_8x = scaleModel(GPT_4, 1 / 8);
export const GPT_4_16x = scaleModel(GPT_4, 1 / 16);

export const Llama_3_gpt4_size = scaleModel(Llama_3_405B, 4.444, 0);
export const Llama_3_gpt4_size_2x = scaleModel(Llama_3_gpt4_size, 1 / 2);
export const Llama_3_gpt4_size_4x = scaleModel(Llama_3_gpt4_size, 1 / 4);
export const Llama_3_gpt4_size_8x = scaleModel(Llama_3_gpt4_size, 1 / 8);
export const Llama_3_gpt4_size_16x = scaleModel(Llama_3_gpt4_size, 1 / 16);

// ---------------------------------------------------------------------------
// Lookup map for UI
// ---------------------------------------------------------------------------

export const MODEL_MAP: Record<string, Model> = {
  "Llama 3 70B": Llama_3_70B,
  "Llama 3 8B": Llama_3_8B,
  "Llama 3 405B": Llama_3_405B,
  "DeepSeek V3": DeepSeek_V3,
  "GPT-4": GPT_4,
  "Mixtral 8x22B": Mixtral_8x22B,
  "Mixtral 8x7B": Mixtral_8x7B,
  "Mistral Large 2": Mistral_Large_2,
};
