//predict/[week]/route
import { NextRequest, NextResponse } from 'next/server';

export async function POST(
    req: NextRequest,
    { params }: { params: { week: string } }
) {
    // parse & validate the week
    const wk = Number(params.week);
    if (![6, 12, 18, 24].includes(wk)) {
        return NextResponse.json(
            { message: `Invalid week: ${params.week}` },
            { status: 400 }
        );
    }

    // parse the incoming body and pull out features
    let body: any;
    try {
        body = await req.json();
    } catch {
        return NextResponse.json(
            { message: "Invalid JSON payload" },
            { status: 400 }
        );
    }
    if (typeof body.features !== "object" || body.features === null) {
        return NextResponse.json(
            { message: "Request must have a top-level `features` object" },
            { status: 400 }
        );
    }

    // proxy to your FastAPI server
    const upstream = await fetch(
        `http://localhost:8000/predict/${wk}`,   // <-- raw number in URL
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: body.features }),
        }
    );

    // 4) handle upstream errors
    if (!upstream.ok) {
        let text: string;
        try {
            text = await upstream.text();
        } catch {
            text = upstream.statusText;
        }
        return NextResponse.json(
            { message: `Upstream error: ${text}` },
            { status: 502 }
        );
    }

    // 5) return whatever FastAPI gave us (prediction + explanations)
    const payload = await upstream.json();
    return NextResponse.json(payload);
}