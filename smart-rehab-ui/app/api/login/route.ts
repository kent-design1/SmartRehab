import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function POST(req: NextRequest) {
    const { email, password } = await req.json();
    // Compare against env vars
    if (
        email === process.env.SHARED_USER &&
        password === process.env.SHARED_PW
    ) {
        const res = NextResponse.json({ success: true });
        res.cookies.set('auth', 'true', {
            httpOnly: true,
            path: '/',
            sameSite: 'lax',
            secure: process.env.NODE_ENV === 'production',
        });
        return res;
    }
    return NextResponse.json({ message: 'Invalid credentials' }, { status: 401 });
}