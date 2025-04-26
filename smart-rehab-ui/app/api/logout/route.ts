
import { NextResponse } from 'next/server';

export async function POST() {
    // Build a JSON 200 response
    const res = NextResponse.json({ success: true });

    // Delete the 'auth' cookie (no options needed)
    res.cookies.delete('auth');

    return res;
}