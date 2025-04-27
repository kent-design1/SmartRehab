// lib/api.ts
export async function getPrediction(
    week: number,
    features: Record<string, any>
): Promise<number> {
    const res = await fetch(`/api/predict/${week}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
    });

    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
            `Prediction failed (${res.status}): ${body.message || res.statusText}`
        );
    }

    const data = await res.json();
    return data.prediction;
}