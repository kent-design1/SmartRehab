'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { Menu, X, User } from 'lucide-react';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'motion/react';
import { cn } from '@/lib/utils';

const LINKS = [
    { name: 'Predict', href: '/predict' },
    { name: 'Dashboard', href: '/dashboard' },
    { name: 'About', href: '/about' },
    { name: 'Contact', href: '/contact' },
    { name: 'API Docs', href: '/apidocs' },
];

export default function Nav() {
    const router = useRouter();
    const [auth, setAuth] = useState(false);
    const [loading, setLoading] = useState(true);
    const [menuOpen, setMenuOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);
    const { scrollY } = useScroll({ target: ref, offset: ['start start', 'end start'] });
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        fetch('/api/auth/session')
            .then(r => r.json())
            .then(d => {
                console.log('Auth session:', d);
                setAuth(!!d.authenticated);
                setLoading(false);
            })
            .catch(() => {
                setAuth(false);
                setLoading(false);
            });
    }, []);

    useMotionValueEvent(scrollY, 'change', y => setScrolled(y > 50));

    const logout = async () => {
        await fetch('/api/logout', { method: 'POST' });
        router.push('/auth/login');
    };

    return (
        <motion.nav
            ref={ref}
            className={cn(
                'fixed inset-x-0 top-0 z-50 transition-all',
                scrolled ? 'backdrop-blur-md bg-white/60 shadow-md' : 'bg-transparent'
            )}
        >
            <div className="mx-auto max-w-7xl flex items-center justify-between px-6 py-4 lg:px-8">
                <Link href="/" className="text-2xl font-extrabold text-gray-800">
                    SmartRehab
                </Link>

                <div className="hidden lg:flex space-x-8 font-black">
                    {LINKS.map(l => (
                        <Link
                            key={l.href}
                            href={l.href}
                            className="text-gray-800 hover:text-blue-500 font-black transition"
                        >
                            {l.name}
                        </Link>
                    ))}
                </div>

                <div className="flex items-center space-x-4 lg:space-x-6">
                    {loading ? null : auth ? (
                        <>
                            <div className="hidden lg:flex items-center justify-center w-8 h-8 bg-white/20 rounded-full">
                                <User className="w-5 h-5 text-gray-800" />
                            </div>
                            <button
                                onClick={logout}
                                className="hidden lg:inline-block bg-red-500 hover:bg-red-600 text-white py-1.5 px-4 rounded-full shadow transition"
                            >
                                Sign Out
                            </button>
                        </>
                    ) : (
                        <Link
                            href="/auth/login"
                            className="hidden lg:inline-block bg-blue-500 hover:bg-blue-600 text-white py-1.5 px-4 rounded-full shadow transition"
                        >
                            Sign In
                        </Link>
                    )}

                    <button
                        onClick={() => setMenuOpen(o => !o)}
                        className="lg:hidden p-2 text-gray-800"
                    >
                        {menuOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                </div>
            </div>

            <AnimatePresence>
                {menuOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden bg-white shadow-md lg:hidden"
                    >
                        <div className="flex flex-col px-6 py-4 space-y-4">
                            {LINKS.map(l => (
                                <Link
                                    key={l.href}
                                    href={l.href}
                                    onClick={() => setMenuOpen(false)}
                                    className="text-gray-800 hover:text-blue-500 font-medium drop-shadow-lg"
                                >
                                    {l.name}
                                </Link>
                            ))}
                            {loading ? null : auth ? (
                                <button
                                    onClick={() => {
                                        logout();
                                        setMenuOpen(false);
                                    }}
                                    className="mt-2 bg-red-500 text-white py-2 rounded-full"
                                >
                                    Sign Out
                                </button>
                            ) : (
                                <Link
                                    href="/auth/login"
                                    onClick={() => setMenuOpen(false)}
                                    className="mt-2 bg-blue-500 text-white py-2 rounded-full text-center"
                                >
                                    Sign In
                                </Link>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.nav>
    );
}