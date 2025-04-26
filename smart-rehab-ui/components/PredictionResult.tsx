interface PredictionResultProps {
    week: number;
    value: number;
}

export default function PredictionResult({ week, value }: PredictionResultProps) {
    return (
        <div className="mt-6 p-4 bg-white rounded-lg shadow-sm border">
            <h2 className="text-lg font-semibold">Predicted SCIM at Week {week}</h2>
            <p className="text-2xl mt-2">{value.toFixed(2)}</p>
        </div> )
}