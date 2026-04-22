import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  return handleProxy(req);
}

export async function GET(req: NextRequest) {
  return handleProxy(req);
}

export async function PUT(req: NextRequest) {
  return handleProxy(req);
}

export async function DELETE(req: NextRequest) {
  return handleProxy(req);
}

async function handleProxy(req: NextRequest) {
  const url = new URL(req.url);
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  
  // Reconstruct the target path (e.g. "/api/v1/chat")
  const targetPath = url.pathname + url.search;
  
  // Forward cookies for NextAuth
  const headers = new Headers();
  const cookieHeader = req.headers.get("cookie");
  if (cookieHeader) {
    headers.set("cookie", cookieHeader);
  }
  
  // Forward content-type
  const contentType = req.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }

  const reqBody = (req.method !== "GET" && req.method !== "HEAD") ? await req.clone().arrayBuffer() : undefined;

  try {
    const response = await fetch(`${backendUrl}${targetPath}`, {
      method: req.method,
      headers: headers,
      body: reqBody,
    });
    
    // Convert fetch response to NextResponse
    const responseData = await response.arrayBuffer();
    const responseHeaders = new Headers(response.headers);
    responseHeaders.delete("content-encoding"); // Let Next.js handle encoding safely
    responseHeaders.delete("content-length"); // Prevent length mismatches
    
    return new NextResponse(responseData, {
      status: response.status,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("Route Handler Proxy error:", error);
    return NextResponse.json({ error: "Internal Proxy Error" }, { status: 504 });
  }
}
