import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Regex to skip static files and the login page
const PUBLIC_FILE = /\.(?:png|jpg|jpeg|css|js|ico|svg)$/;

export function middleware(req: NextRequest) {
    const { pathname } = req.nextUrl;

    // Allow access to root, login, API routes, and static assets
    if (
        pathname === '/' ||
        pathname === '/auth/login' ||
        pathname.startsWith('/api/login') ||
        pathname.startsWith('/api/logout') ||
        PUBLIC_FILE.test(pathname)
    ) {
        return NextResponse.next();
    }

    // Check auth cookie
    const auth = req.cookies.get('auth')?.value;
    if (auth === 'true') {
        return NextResponse.next();
    }

    // Redirect to login
    const loginUrl = req.nextUrl.clone();
    loginUrl.pathname = '/auth/login';
    return NextResponse.redirect(loginUrl);
}

export const config = {
    matcher: ['/:path*'],
};
