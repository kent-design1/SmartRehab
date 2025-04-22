// "use client";
//
// import Image from "next/image";
// import { useRouter } from "next/navigation";
// import {HeroBg} from "@/assets";
//
// export default function WelcomePage() {
//     const router = useRouter();
//
//     return (
//         <div className="relative flex flex-col items-center justify-center min-h-screen">
//             {/* Background image covering whole container */}
//             <Image
//                 src={HeroBg}
//                 fill
//                 className="object-cover"
//                 alt="Doctor and Patient"
//             />
//             {/* Optional overlay for better text contrast */}
//             <div className="absolute inset-0 bg-black opacity-30"></div>
//
//             {/* Content container */}
//             <div className="relative z-10 text-center px-4">
//                 <h1 className="text-[5rem] font-extrabold text-white mt-6">
//                     Welcome to RehabPredict
//                 </h1>
//                 <p className="text-3xl text-white mt-2">
//                     AI-powered rehabilitation insights for spinal injury recovery.
//                 </p>
//                 <button
//                     onClick={() => router.push("/auth/signin")}
//                     className="mt-6 px-6 py-3 bg-blue-600 text-white text-lg font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-300"
//                 >
//                     Sign In
//                 </button>
//             </div>
//         </div>
//     );
// }

"use client";

import { useState, FormEvent } from "react";

interface HybridPredictionResult {
    "Predicted Week8 SCIM": number;
    "Therapy Recommendation": string;
    "Predicted Total Cost": number;
    "Cost Efficiency": number;
}

// Predefined options for dropdowns
const genderOptions = ["Male", "Female"];
const conditionOptions = ["Stroke Recovery", "Orthopedic Injury", "Chronic Pain", "Diabetes", "Hypertension"];
const therapyPlanOptions = ["Physiotherapy", "Occupational Therapy", "Medication", "Lifestyle Changes", "Comprehensive SCI Rehab"];

/**
 * One-hot encodes a given value based on the provided options.
 * For example, if options = ["Male", "Female"] and value = "Female",
 * returns [0, 1].
 */
function oneHotEncode(value: string, options: string[]): number[] {
    return options.map((option) => (option === value ? 1 : 0));
}

/**
 * Constructs the baseline feature vector.
 * Order: [Baseline SCIM, Total Therapy Cost, Age] +
 *        one-hot(Gender) (2 features) +
 *        one-hot(Condition) (5 features) +
 *        one-hot(Therapy Plan) (5 features).
 * Total features: 3 + 2 + 5 + 5 = 15.
 */
function constructBaselineVector(
    baselineScim: number,
    totalCost: number,
    age: number,
    gender: string,
    condition: string,
    therapyPlan: string
): number[] {
    const numericPart: number[] = [baselineScim, totalCost, age];
    const genderEncoded: number[] = oneHotEncode(gender, genderOptions);
    const conditionEncoded: number[] = oneHotEncode(condition, conditionOptions);
    const therapyPlanEncoded: number[] = oneHotEncode(therapyPlan, therapyPlanOptions);

    return [...numericPart, ...genderEncoded, ...conditionEncoded, ...therapyPlanEncoded];
}

export default function Home() {
    // State for baseline inputs
    const [baselineScim, setBaselineScim] = useState<string>("");
    const [totalCost, setTotalCost] = useState<string>("");
    const [age, setAge] = useState<string>("");

    // State for categorical choices
    const [gender, setGender] = useState<string>(genderOptions[0]);
    const [condition, setCondition] = useState<string>(conditionOptions[0]);
    const [therapyPlan, setTherapyPlan] = useState<string>(therapyPlanOptions[0]);

    // State for weekly scores (comma-separated)
    const [weeklyScores, setWeeklyScores] = useState<string>("");

    // State for result and error messages
    const [result, setResult] = useState<HybridPredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setError(null);
        setResult(null);

        // Basic validation: Check required fields
        if (!baselineScim || !totalCost || !age) {
            setError("Please provide Baseline SCIM, Total Therapy Cost, and Age.");
            return;
        }

        // Construct the baseline vector from multiple fields
        const baselineVector = constructBaselineVector(
            parseFloat(baselineScim),
            parseFloat(totalCost),
            parseFloat(age),
            gender,
            condition,
            therapyPlan
        );

        // Parse weekly SCIM scores input; expects comma-separated values for 18 weeks.
        const weeklyArray =
            weeklyScores.trim() !== ""
                ? weeklyScores.split(",").map((s) => parseFloat(s.trim()))
                : null;

        // Prepare the payload to send to the Flask API.
        const payload = {
            baseline: baselineVector,
            weekly: weeklyArray,
            alpha: 0.5, // Adjust the blending weight as needed.
        };

        try {
            const res = await fetch("http://localhost:5001/api/calculate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await res.json();

            if (!res.ok) {
                setError(data.error || "Error occurred");
            } else {
                setResult(data);
            }
        } catch (err) {
            console.error(err);
            setError("Error connecting to server");
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
            <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 max-w-2xl">
                <h1 className="text-2xl font-bold text-center mb-6">
                    Smart Rehab Prediction
                </h1>
                <form onSubmit={handleSubmit}>
                    {/* Baseline SCIM Input */}
                    <div className="mb-4">
                        <label className="block text-gray-700 text-sm font-bold mb-1">
                            Baseline SCIM Score
                        </label>
                        <input
                            type="number"
                            value={baselineScim}
                            onChange={(e) => setBaselineScim(e.target.value)}
                            placeholder="e.g., 45"
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
                            required
                        />
                    </div>
                    {/* Total Therapy Cost Input */}
                    <div className="mb-4">
                        <label className="block text-gray-700 text-sm font-bold mb-1">
                            Total Therapy Cost
                        </label>
                        <input
                            type="number"
                            value={totalCost}
                            onChange={(e) => setTotalCost(e.target.value)}
                            placeholder="e.g., 5000"
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
                            required
                        />
                    </div>
                    {/* Age Input */}
                    <div className="mb-4">
                        <label className="block text-gray-700 text-sm font-bold mb-1">
                            Age
                        </label>
                        <input
                            type="number"
                            value={age}
                            onChange={(e) => setAge(e.target.value)}
                            placeholder="e.g., 50"
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
                            required
                        />
                    </div>
                    {/* Gender Select */}
                    <div className="mb-4">
                        <label className="block text-gray-700 text-sm font-bold mb-1">
                            Gender
                        </label>
                        <select
                            value={gender}
                            onChange={(e) => setGender(e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
                        >
                            {genderOptions.map((option) => (
                                <option key={option} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                    {/* Condition Select */}
                    <div className="mb-4">
                        <label className="block text-gray-700 text-sm font-bold mb-1">
                            Condition
                        </label>
                        <select
                            value={condition}
                            onChange={(e) => setCondition(e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
                        >
                            {conditionOptions.map((option) => (
                                <option key={option} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                    {/* Therapy Plan Select */}
                    <div className="mb-6">
                        <label className="block text-gray-700 text-sm font-bold mb-1">
                            Therapy Plan
                        </label>
                        <select
                            value={therapyPlan}
                            onChange={(e) => setTherapyPlan(e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
                        >
                            {therapyPlanOptions.map((option) => (
                                <option key={option} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                    {/* Weekly SCIM Scores Input */}
                    <div className="mb-6">
                        <label className="block text-gray-700 text-sm font-bold mb-1">
                            Weekly SCIM Scores (Weeks 0-17)
                        </label>
                        <input
                            type="text"
                            value={weeklyScores}
                            onChange={(e) => setWeeklyScores(e.target.value)}
                            placeholder="e.g., 45,47,46,48,..."
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
                        />
                        <p className="text-xs text-gray-500 mt-1">
                            (Enter 18 comma-separated values; leave empty to use static model only)
                        </p>
                    </div>
                    <div className="flex items-center justify-between">
                        <button
                            type="submit"
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        >
                            Predict
                        </button>
                    </div>
                </form>
                {error && <p className="mt-4 text-red-500">{error}</p>}
                {result && (
                    <div className="mt-4 space-y-2">
                        <p>
                            <span className="font-bold">Predicted Week8 SCIM:</span>{" "}
                            {result["Predicted Week8 SCIM"].toFixed(2)}
                        </p>
                        <p>
                            <span className="font-bold">Therapy Recommendation:</span>{" "}
                            {result["Therapy Recommendation"]}
                        </p>
                        <p>
                            <span className="font-bold">Predicted Total Cost:</span>{" "}
                            {result["Predicted Total Cost"]}
                        </p>
                        <p>
                            <span className="font-bold">Cost Efficiency:</span>{" "}
                            {result["Cost Efficiency"].toFixed(2)}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}