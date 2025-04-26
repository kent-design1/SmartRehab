// components/Navbar.tsx
import Link from "next/link";

export default function Navbar() {
    return (
        <nav className="fixed top-0 left-0 w-full bg-white/30 backdrop-blur-md shadow-lg py-4 px-6 flex items-center z-50">
            {/* Logo */}
            <div className="flex-shrink-0">
                <Link href="/" className="text-2xl font-extrabold text-white drop-shadow-lg">
                    SmartRehab
                </Link>
            </div>

            {/* Main nav links */}
            <div className="flex-grow ml-10 space-x-8">
                <Link href="/predict" className="text-white hover:text-blue-200 transition font-medium">
                    Predict
                </Link>
                <Link href="/dashboard" className="text-white hover:text-blue-200 transition font-medium">
                    Dashboard
                </Link>
                <Link href="/about" className="text-white hover:text-blue-200 transition font-medium">
                    About
                </Link>
                <Link href="/contact" className="text-white hover:text-blue-200 transition font-medium">
                    Contact
                </Link>
                <Link href="/docs" className="text-white hover:text-blue-200 transition text-sm font-medium">
                    API Docs
                </Link>
            </div>

            {/* Auth buttons */}
            <div className="flex-shrink-0 space-x-4">
                <Link
                    href="/auth/login"
                    className="text-white hover:text-blue-200 transition font-medium"
                >
                    Login
                </Link>
                <Link
                    href="/auth/signup"
                    className="bg-blue-500 hover:bg-blue-600 text-white py-1.5 px-4 rounded-full shadow transition"
                >
                    Sign Up
                </Link>
            </div>
        </nav>
    );
}