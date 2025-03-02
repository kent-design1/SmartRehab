"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import {HeroBg} from "@/assets";

export default function WelcomePage() {
    const router = useRouter();

    return (
        <div className="relative flex flex-col items-center justify-center min-h-screen">
            {/* Background image covering whole container */}
            <Image
                src={HeroBg}
                fill
                className="object-cover"
                alt="Doctor and Patient"
            />
            {/* Optional overlay for better text contrast */}
            <div className="absolute inset-0 bg-black opacity-30"></div>

            {/* Content container */}
            <div className="relative z-10 text-center px-4">
                <h1 className="text-[5rem] font-extrabold text-white mt-6">
                    Welcome to RehabPredict
                </h1>
                <p className="text-3xl text-white mt-2">
                    AI-powered rehabilitation insights for spinal injury recovery.
                </p>
                <button
                    onClick={() => router.push("/auth/signin")}
                    className="mt-6 px-6 py-3 bg-blue-600 text-white text-lg font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-300"
                >
                    Sign In
                </button>
            </div>
        </div>
    );
}