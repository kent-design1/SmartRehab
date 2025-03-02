"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function AuthHomePage() {
    const router = useRouter();

    useEffect(() => {
        router.push("/auth/Welcome");
    }, [router]);

    return <p className="text-center mt-10 text-gray-600">Loading...</p>;
}