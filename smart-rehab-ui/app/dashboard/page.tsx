// app/dashboard/route.ts
'use client';
import { ChartPie, TrendingUp, BarChart } from 'lucide-react';

export default function DashboardPage() {
    const metrics = [
        { week: 6, rmse: 1.37, mae: 1.03 },
        { week: 12, rmse: 1.65, mae: 1.31 },
        { week: 18, rmse: 1.86, mae: 1.49 },
        { week: 24, rmse: 2.25, mae: 1.79 },
    ];

    const avgRmse = (metrics.reduce((sum, m) => sum + m.rmse, 0) / metrics.length).toFixed(2);
    const avgMae = (metrics.reduce((sum, m) => sum + m.mae, 0) / metrics.length).toFixed(2);

    return (
        <main className="min-h-screen bg-gray-50 py-16 px-6 lg:px-24">
            <h1 className="text-4xl font-extrabold text-gray-900 mb-12">Dashboard</h1>

            {/* Performance Bento Grid */}
            <section className="grid grid-cols-1 lg:grid-cols-5 grid-rows-2 lg:grid-rows-1 gap-6 mb-16">
                {/* Overall Performance */}
                <div className="lg:col-span-3 row-span-1 bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-3xl shadow-xl p-8 flex flex-col justify-center transition transform hover:scale-105">
                    <div className="flex items-center mb-4">
                        <ChartPie className="w-8 h-8 mr-2" />
                        <h2 className="text-2xl font-semibold">Overall Performance</h2>
                    </div>
                    <p className="text-lg">Average RMSE: <span className="font-bold">{avgRmse}</span></p>
                    <p className="text-lg">Average MAE: <span className="font-bold">{avgMae}</span></p>
                </div>

                {/* Week-specific metrics */}
                {metrics.map((m) => (
                    <div
                        key={m.week}
                        className="bg-white rounded-2xl shadow-lg p-6 flex flex-col justify-between transition transform hover:scale-105 hover:shadow-2xl"
                    >
                        <div className="flex items-center mb-4">
                            <TrendingUp className="w-6 h-6 text-blue-500 mr-2" />
                            <h3 className="text-xl font-semibold">Week {m.week}</h3>
                        </div>
                        <div className="space-y-1">
                            <p className="text-gray-600">
                                RMSE: <span className="font-medium text-gray-800">{m.rmse}</span>
                            </p>
                            <p className="text-gray-600">
                                MAE: <span className="font-medium text-gray-800">{m.mae}</span>
                            </p>
                        </div>
                    </div>
                ))}
            </section>

            {/* Charts Section */}
            <section>
                <h2 className="text-2xl font-semibold text-gray-900 mb-6">Error Analysis</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white rounded-3xl shadow-xl p-6 transition hover:shadow-2xl">
                        <div className="flex items-center mb-4">
                            <BarChart className="w-6 h-6 text-indigo-500 mr-2" />
                            <h3 className="text-lg font-semibold">Residuals Histogram</h3>
                        </div>
                        <div className="h-64 bg-gray-200 rounded-lg flex items-center justify-center text-gray-500">
                            Chart Placeholder
                        </div>
                    </div>

                    <div className="bg-white rounded-3xl shadow-xl p-6 transition hover:shadow-2xl">
                        <div className="flex items-center mb-4">
                            <BarChart className="w-6 h-6 text-indigo-500 mr-2" />
                            <h3 className="text-lg font-semibold">Residuals vs Predicted</h3>
                        </div>
                        <div className="h-64 bg-gray-200 rounded-lg flex items-center justify-center text-gray-500">
                            Chart Placeholder
                        </div>
                    </div>
                </div>
            </section>
        </main>
    );
}