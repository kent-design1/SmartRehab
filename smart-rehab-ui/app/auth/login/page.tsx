// app/auth/login/page.tsx
'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Hospital, Eye, EyeOff } from 'lucide-react';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPwd, setShowPwd] = useState(false);
    const [error, setError] = useState('');
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const res = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });
        if (res.ok) {
            router.replace('/predict');
        } else {
            const data = await res.json();
            setError(data.message || 'Login failed');
        }
    };

    return (
        <div className="flex items-center justify-center h-screen bg-gradient-to-br from-sky-100 via-white to-blue-200">
            <form
                onSubmit={handleSubmit}
                className="relative bg-white bg-opacity-90 backdrop-blur-md p-8 rounded-2xl shadow-2xl w-full max-w-md"
            >
                {/* Hospital Icon */}
                <div className="flex justify-center mb-6">
                    <Hospital className="w-12 h-12 text-blue-600" />
                </div>

                <h2 className="text-3xl font-extrabold text-center text-blue-700 mb-4">
                    Doctor Login
                </h2>

                {error && (
                    <p className="text-center text-red-600 mb-4">{error}</p>
                )}

                <div className="space-y-4">
                    <div>
                        <label className="block text-gray-700 font-medium mb-1">
                            Hospital Email
                        </label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className="w-full border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg p-3 transition"
                        />
                    </div>

                    <div className="relative">
                        <label className="block text-gray-700 font-medium mb-1">
                            Password
                        </label>
                        <input
                            type={showPwd ? 'text' : 'password'}
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="w-full border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg p-3 pr-10 transition"
                        />
                        <button
                            type="button"
                            onClick={() => setShowPwd((v) => !v)}
                            className="absolute inset-y-0 top-8 right-3 flex items-center text-gray-500 hover:text-gray-700"
                            aria-label={showPwd ? 'Hide password' : 'Show password'}
                        >
                            {showPwd ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                        </button>
                    </div>
                </div>

                <button
                    type="submit"
                    className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg shadow-lg transition transform hover:-translate-y-0.5"
                >
                    Log In
                </button>
            </form>
        </div>
    );
}