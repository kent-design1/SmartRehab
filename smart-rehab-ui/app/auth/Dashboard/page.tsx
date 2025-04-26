"use client";

import { useState } from "react";
import { Line, Pie } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement
);

interface PredictionResult {
    predictedWeek8SCIM: number;
    therapyRecommendation: string;
    predictedTotalCost: number;
    costEfficiency: number;
}

// Dummy Data for Demonstration
const dummyPrediction: PredictionResult = {
    predictedWeek8SCIM: 73.45,
    therapyRecommendation: "Increase Frequency",
    predictedTotalCost: 4500,
    costEfficiency: 0.016,
};

// Dummy weekly SCIM scores over 18 weeks
const dummyWeeklyScores: number[] = [
    45, 47, 48, 50, 51, 52, 54, 55, 57, 58, 59, 60, 62, 63, 64, 66, 67, 68,
];

// Dummy Therapy Recommendation Distribution Data (for pie chart)
const dummyTherapyDist = {
    labels: ["Change Therapy", "Increase Frequency", "Maintain Plan"],
    datasets: [
        {
            data: [25, 50, 25],
            backgroundColor: ["#ef4444", "#f59e0b", "#10b981"],
            hoverBackgroundColor: ["#dc2626", "#d97706", "#059669"],
        },
    ],
};

export default function Dashboard() {
    // In a real app, these values are obtained from the Flask API.
    const [prediction] = useState<PredictionResult>(dummyPrediction);
    const [weeklyScores] = useState<number[]>(dummyWeeklyScores);

    // Line Chart: Weekly SCIM Scores
    const lineChartData = {
        labels: weeklyScores.map((_, i) => `Week ${i}`),
        datasets: [
            {
                label: "Weekly SCIM Score",
                data: weeklyScores,
                fill: false,
                borderColor: "#3b82f6",
                backgroundColor: "#bfdbfe",
                tension: 0.2,
            },
        ],
    };

    // Pie Chart: Therapy Recommendation Distribution
    const pieChartData = dummyTherapyDist;

    return (
        <div className="min-h-screen bg-gradient-to-r from-gray-100 to-blue-50 p-8">
            {/* Hero Section */}
            <header className="mb-10 text-center">
                <h1 className="text-5xl font-extrabold text-gray-800 mb-4">
                    Smart Rehab Dashboard
                </h1>
                <p className="text-xl text-gray-600">
                    Unified Prediction Model for Personalized Spinal Rehabilitation
                </p>
            </header>

            {/* Summary Cards */}
            <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
                <div className="bg-white rounded-xl shadow-lg p-6 flex flex-col items-center">
                    <p className="text-gray-500 uppercase text-sm font-semibold">Week8 SCIM</p>
                    <p className="text-3xl font-bold text-blue-600 mt-2">
                        {prediction.predictedWeek8SCIM.toFixed(2)}
                    </p>
                </div>
                <div className="bg-white rounded-xl shadow-lg p-6 flex flex-col items-center">
                    <p className="text-gray-500 uppercase text-sm font-semibold">
                        Therapy Rec.
                    </p>
                    <p className="text-2xl font-bold text-green-600 mt-2">
                        {prediction.therapyRecommendation}
                    </p>
                </div>
                <div className="bg-white rounded-xl shadow-lg p-6 flex flex-col items-center">
                    <p className="text-gray-500 uppercase text-sm font-semibold">
                        Total Cost
                    </p>
                    <p className="text-3xl font-bold text-purple-600 mt-2">
                        ${prediction.predictedTotalCost}
                    </p>
                </div>
                <div className="bg-white rounded-xl shadow-lg p-6 flex flex-col items-center">
                    <p className="text-gray-500 uppercase text-sm font-semibold">
                        Cost Efficiency
                    </p>
                    <p className="text-3xl font-bold text-red-600 mt-2">
                        {prediction.costEfficiency.toFixed(3)}
                    </p>
                </div>
            </section>

            {/* Detailed Visualizations */}
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Weekly SCIM Trend Line Chart */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                    <h2 className="text-2xl font-semibold mb-4">Weekly SCIM Trend</h2>
                    <Line data={lineChartData} options={{ responsive: true, plugins: { legend: { position: "top" } } }} />
                </div>
                {/* Therapy Recommendation Distribution Pie Chart */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                    <h2 className="text-2xl font-semibold mb-4">
                        Therapy Recommendation Distribution
                    </h2>
                    <Pie data={pieChartData} options={{ responsive: true, plugins: { legend: { position: "bottom" } } }} />
                </div>
            </section>

            {/* Explanation Card */}
            <section className="mt-12 bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-2xl font-semibold mb-4">Overview</h2>
                <p className="text-gray-700 leading-relaxed">
                    This dashboard showcases the output of our unified prediction model for spinal
                    rehabilitation. The model integrates both static data (such as patient demographics,
                    baseline SCIM, and therapy details) and dynamic time-series data (weekly SCIM scores) to
                    generate personalized predictions for Week 8 outcomes, therapy recommendations, and cost efficiency.
                    The visualizations illustrate the weekly trend in SCIM scores and the distribution of therapy
                    recommendations across our dataset.
                </p>
            </section>
        </div>
    );
}