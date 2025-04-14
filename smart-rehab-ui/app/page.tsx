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

interface CalculationResult {
    input: number;
    result: number;
}

export default function Home() {
    // State types
    const [number, setNumber] = useState<string>("");
    const [result, setResult] = useState<CalculationResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setError(null);
        setResult(null);

        try {
            // Update URL as needed; ensure this matches your Flask server port.
            const res = await fetch("http://localhost:5001/api/calculate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ number: number }),
            });

            const data = await res.json();

            if (!res.ok) {
                setError(data.error || "Error occurred");
            } else {
                setResult(data);
            }
        } catch (err) {
            setError("Error connecting to server");
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4">
            <div className="w-full max-w-md bg-white shadow-md rounded px-8 pt-6 pb-8">
                <h1 className="text-2xl font-bold text-center mb-6">
                    Smart Rehab Calculator
                </h1>
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label
                            htmlFor="numberInput"
                            className="block text-gray-700 text-sm font-bold mb-2"
                        >
                            Enter a number:
                        </label>
                        <input
                            id="numberInput"
                            type="number"
                            value={number}
                            onChange={(e) => setNumber(e.target.value)}
                            placeholder="e.g., 5"
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        />
                    </div>
                    <div className="flex items-center justify-between">
                        <button
                            type="submit"
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        >
                            Calculate
                        </button>
                    </div>
                </form>
                {error && <p className="mt-4 text-red-500">{error}</p>}
                {result && (
                    <div className="mt-4">
                        <p className="text-lg">
                            Input Number: <span className="font-bold">{result.input}</span>
                        </p>
                        <p className="text-lg">
                            Calculated Result (Square):{" "}
                            <span className="font-bold">{result.result}</span>
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}