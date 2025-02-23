import Link from "next/link";

export default function Navbar() {
    return (
        <nav className="w-full bg-gradient-to-r from-blue-600 to-indigo-700 shadow-xl py-4 px-8 flex justify-between items-center fixed top-0 z-50">
            <h1 className="text-3xl font-extrabold text-white drop-shadow-lg">
                Smart Rehab
            </h1>
            <div className="space-x-8">
                <Link href="/" legacyBehavior>
                    <a className="text-lg font-medium text-white hover:text-yellow-300 transition-all duration-300 ease-in-out">
                        Home
                    </a>
                </Link>
                <Link href="/predict" legacyBehavior>
                    <a className="text-lg font-medium text-white hover:text-yellow-300 transition-all duration-300 ease-in-out">
                        Prediction
                    </a>
                </Link>
                <Link href="/about" legacyBehavior>
                    <a className="text-lg font-medium text-white hover:text-yellow-300 transition-all duration-300 ease-in-out">
                        About
                    </a>
                </Link>
                <Link href="/contact" legacyBehavior>
                    <a className="text-lg font-medium text-white hover:text-yellow-300 transition-all duration-300 ease-in-out">
                        Contact
                    </a>
                </Link>
            </div>
        </nav>
    );
}