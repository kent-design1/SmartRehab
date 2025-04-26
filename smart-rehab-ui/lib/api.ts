export async function getPrediction(
    week: number,
    features: Record<string, any>
): Promise<number> {
    const res = await fetch(`/predict/week${week}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    return data.prediction;
}