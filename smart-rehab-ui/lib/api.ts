export interface CostExplanation {
    feature: string;
    recommendation: string;
}

export interface CostBreakdown {
    baseline_cost: number;
    overuse_cost: number;
    total_cost: number;
    efficiency: number | null;
    explanations: CostExplanation[];
}

export interface StaticRecommendation {
    feature: string;
    recommendation: string;
    rationale: string;
}

export interface ShapRecommendation {
    feature: string;
    shap_value: number;
    recommendation: string;
    rationale: string;
}

export interface PredictionResponse {
    week: number;
    prediction: number;
    cost: CostBreakdown;
    static_recommendations: StaticRecommendation[];
    shap_recommendations: ShapRecommendation[];
}

export async function getPrediction(
    week: number,
    features: Record<string, any>
): Promise<PredictionResponse> {
    const url = `/api/predict/${week}`;
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
    });

    if (!res.ok) {
        let body: any = {};
        try { body = await res.json(); } catch {}
        throw new Error(
            `Prediction failed (${res.status}): ${body.message ?? res.statusText}`
        );
    }

    const data = (await res.json()) as PredictionResponse;
    if (typeof data.prediction !== 'number') {
        throw new Error('Invalid response: missing numeric prediction');
    }
    return data;
}
