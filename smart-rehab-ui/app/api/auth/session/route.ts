// File: app/api/auth/session/route.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function GET(req: NextRequest) {
    const authCookie = req.cookies.get('auth')?.value;
    const isAuthenticated = authCookie === 'true';
    return NextResponse.json({ authenticated: isAuthenticated });
}