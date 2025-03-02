import Link from "next/link";

export default function Footer() {
    return (
        <footer className="w-full bg-gradient-to-r from-indigo-700 to-blue-600 text-white py-4 px-8  bottom-0 z-50">
            <div className="flex justify-between items-center">
                <p className="text-sm">
                    © {new Date().getFullYear()} Smart Rehab. All rights reserved.
                </p>
                <div className="flex space-x-4">
                    <Link href="/about" legacyBehavior>
                        <a className="text-sm hover:text-yellow-300 transition-all duration-300 ease-in-out">
                            About
                        </a>
                    </Link>
                    <Link href="/contact" legacyBehavior>
                        <a className="text-sm hover:text-yellow-300 transition-all duration-300 ease-in-out">
                            Contact
                        </a>
                    </Link>
                    <Link href="/privacy" legacyBehavior>
                        <a className="text-sm hover:text-yellow-300 transition-all duration-300 ease-in-out">
                            Privacy Policy
                        </a>
                    </Link>
                </div>
            </div>
        </footer>
    );
}