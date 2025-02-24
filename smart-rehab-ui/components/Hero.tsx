import Link from "next/link";


export default function Hero() {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
            <div className="bg-white shadow-lg rounded-2xl p-8 max-w-lg text-center mt-10">
                <h1 className="text-4xl font-bold text-blue-600 mb-4">Welcome to Smart Rehab</h1>
                <p className="text-gray-600 text-lg mb-6">AI-Powered Spinal Injury Recovery Predictor</p>

                <div className="flex space-x-4">
                    <Link href="/predict">
                        <button className="bg-blue-500 text-white px-6 py-3 rounded-lg shadow-md hover:bg-blue-700 transition duration-300">
                            Get Prediction
                        </button>
                    </Link>
                    <Link href="/about">
                        <button className="bg-gray-300 text-gray-700 px-6 py-3 rounded-lg shadow-md hover:bg-gray-400 transition duration-300">
                            Learn More
                        </button>
                    </Link>
                </div>
            </div>
        </div>
    );
}