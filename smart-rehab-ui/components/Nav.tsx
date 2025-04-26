// components/Navbar.tsx
'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { User } from 'lucide-react';

export default function Navbar() {
    const [authenticated, setAuthenticated] = useState(false);
    const router = useRouter();

    // Check session on mount
    useEffect(() => {
        fetch('/api/auth/session')
            .then(res => res.json())
            .then(data => setAuthenticated(!!data.authenticated))
            .catch(() => setAuthenticated(false));
    }, []);

    const handleLogout = async () => {
        await fetch('/api/logout', { method: 'POST' });
        router.push('/auth/login');
    };

    return (
        <nav className="fixed top-0 left-0 w-full bg-white/30 backdrop-blur-md shadow-lg py-4 px-6 flex items-center z-50">
            {/* Logo */}
            <div className="flex-shrink-0">
                <Link href="/" className="text-2xl font-extrabold text-gray-800 drop-shadow-lg">
                    SmartRehab
                </Link>
            </div>

            {/* Main nav links */}
            <div className="flex-grow ml-10 space-x-8">
                <Link href="/predict" className="text-gray-800 hover:text-blue-200 transition font-medium">
                    Predict
                </Link>
                <Link href="/dashboard" className="text-gray-800 hover:text-blue-200 transition font-medium">
                    Dashboard
                </Link>
                <Link href="/about" className="text-gray-800 hover:text-blue-200 transition font-medium">
                    About
                </Link>
                <Link href="/contact" className="text-gray-800 hover:text-blue-200 transition font-medium">
                    Contact
                </Link>
                <Link href="/docs" className="text-gray-800 hover:text-blue-200 transition text-sm font-medium">
                    API Docs
                </Link>
            </div>

            {/* Auth area */}
            <div className="flex-shrink-0 flex items-center space-x-4">
                {!authenticated ? (
                    <Link
                        href="/auth/login"
                        className="bg-blue-500 hover:bg-blue-600 text-white py-1.5 px-4 rounded-full shadow transition"
                    >
                        Sign In
                    </Link>
                ) : (
                    <>
                        {/* Profile Icon */}
                        <div className="p-1 bg-white/20 backdrop-blur rounded-full">
                            <User className="w-6 h-6 text-white" />
                        </div>
                        {/* Sign Out */}
                        <button
                            onClick={handleLogout}
                            className="bg-red-500 hover:bg-red-600 text-white py-1.5 px-4 rounded-full shadow transition"
                        >
                            Sign Out
                        </button>
                    </>
                )}
            </div>
        </nav>
    );
}