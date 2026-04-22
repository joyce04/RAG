"use client";

import { useEffect, useState, useRef, FormEvent } from "react";
import { signOut, useSession } from "next-auth/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { LogOut, Plus, MessageSquare, ChevronRight, X, User as UserIcon, Bot, FileText, Trash2 } from "lucide-react";

type Reference = {
  source: string;
  page: number;
  snippet: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  references?: Reference[];
};

type Session = {
  id: string;
  title: string;
  created_at: string;
};

export default function ChatDashboard() {
  const { data: sessionData, status } = useSession();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [viewingPdf, setViewingPdf] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch sessions on load
  useEffect(() => {
    if (status === "authenticated") {
      fetchSessions();
    }
  }, [status]);

  // Fetch messages when active session changes
  useEffect(() => {
    if (activeSession) {
      fetchMessages(activeSession);
    } else {
      setMessages([]);
    }
  }, [activeSession]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchSessions = async () => {
    try {
      const res = await fetch("/api/v1/sessions", {
        headers: { "Content-Type": "application/json" },
      });
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const fetchMessages = async (sessionId: string) => {
    try {
      const res = await fetch(`/api/v1/sessions/${sessionId}/messages`, {
        headers: { "Content-Type": "application/json" },
      });
      if (res.ok) {
        const data = await res.json();
        setMessages(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleNewChat = async () => {
    try {
      const res = await fetch("/api/v1/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Chat" }),
      });
      if (res.ok) {
        const newSession = await res.json();
        setSessions([newSession, ...sessions]);
        setActiveSession(newSession.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const deleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const res = await fetch(`/api/v1/sessions/${sessionId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setSessions(prev => prev.filter(s => s.id !== sessionId));
        if (activeSession === sessionId) {
          setActiveSession(null);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  const sendMessage = async (e: FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    let targetSessionId = activeSession;

    // Create new session automatically if none exists
    if (!targetSessionId) {
      try {
        const res = await fetch("/api/v1/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: "New Chat" }),
        });
        if (res.ok) {
          const newSession = await res.json();
          setSessions([newSession, ...sessions]);
          setActiveSession(newSession.id);
          targetSessionId = newSession.id;
        } else {
          return;
        }
      } catch (err) {
        console.error(err);
        return;
      }
    }

    const question = inputValue.trim();
    setInputValue("");
    setMessages((prev) => [
      ...prev,
      { id: Date.now().toString(), role: "user", content: question },
    ]);
    setIsLoading(true);

    try {
      const res = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: targetSessionId,
          question,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setMessages((prev) => [...prev, data]);
        // Refresh sessions to get updated title
        fetchSessions();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const openPdf = (source: string, page: number, snippet?: string) => {
    // Strip absolute paths for legacy or un-processed absolute DB documents
    const filename = source.split(/[/\\]/).pop() || source;
    // Determine the host for API proxying or Next rewrite
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    let hashParams = `page=${page}`;
    if (snippet) {
      hashParams += `&search=${encodeURIComponent(snippet.substring(0, 40))}`;
    }

    setViewingPdf(`${apiUrl}/api/v1/docs/${filename}#${hashParams}`);
  };

  if (status === "loading") {
    return <div className="min-h-screen bg-zinc-950 flex items-center justify-center text-zinc-500">Loading...</div>;
  }

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden font-sans">
      {/* Left Sidebar (Sessions) */}
      <div className="w-64 bg-zinc-900 border-r border-zinc-800 flex flex-col flex-shrink-0 transition-all duration-300">
        <div className="p-4">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 px-4 py-2.5 rounded-lg transition-colors text-sm font-medium"
          >
            <Plus size={18} />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-thin">
          <div className="text-xs font-semibold text-zinc-500 px-3 pt-3 pb-2 uppercase tracking-wider">
            History
          </div>
          {sessions.map((sess) => (
            <button
              key={sess.id}
              onClick={() => setActiveSession(sess.id)}
              className={`group w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-left truncate text-sm transition-colors ${activeSession === sess.id
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
                }`}
            >
              <div className="flex items-center gap-3 truncate">
                <MessageSquare size={16} className="flex-shrink-0" />
                <span className="truncate">{sess.title}</span>
              </div>
              <div
                onClick={(e) => deleteSession(sess.id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
              >
                <Trash2 size={14} />
              </div>
            </button>
          ))}
          {sessions.length === 0 && (
            <div className="px-3 py-4 text-center text-sm text-zinc-600">
              No recent chats
            </div>
          )}
        </div>

        <div className="p-4 border-t border-zinc-800 flex flex-col gap-3">
          <div className="flex items-center gap-3 px-2">
            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center font-bold text-sm">
              {sessionData?.user?.name?.charAt(0) || "U"}
            </div>
            <div className="flex flex-col truncate">
              <span className="text-sm font-medium truncate">
                {sessionData?.user?.name || "User"}
              </span>
            </div>
          </div>
          <button
            onClick={() => signOut()}
            className="text-left flex items-center gap-2 text-zinc-400 hover:text-red-400 text-sm px-2 transition-colors py-1"
          >
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className={`flex flex-col flex-1 relative transition-all duration-300 ${viewingPdf ? "w-1/2 flex-none border-r border-zinc-800" : ""}`}>
        {/* Header */}
        <header className="h-14 flex items-center px-6 border-b border-zinc-800/50 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-10 flex-shrink-0">
          <h2 className="text-zinc-100 font-medium">Adaptive RAG</h2>
          <div className="ml-auto flex items-center gap-4 text-xs text-zinc-500 font-mono">
            Competition Law AI
          </div>
        </header>

        {/* Message List */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 space-y-8">
          {(!messages || messages.length === 0) ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4 max-w-lg mx-auto pb-32">
              <div className="w-16 h-16 bg-zinc-900 rounded-2xl flex items-center justify-center mb-6 border border-zinc-800/50 shadow-sm">
                <Bot size={32} className="text-indigo-400" />
              </div>
              <h1 className="text-2xl font-medium text-zinc-100 mb-3">How can I help you today?</h1>
              <p className="text-zinc-400 text-sm">
                Ask questions about Korean Competition Law Cases. The assistant will search the ingested documents and provide a cited response.
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className="flex flex-col max-w-4xl mx-auto group">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center border shadow-sm">
                    {msg.role === "assistant" ? (
                      <div className="w-full h-full bg-zinc-800 border-zinc-700 flex items-center justify-center rounded-full text-indigo-400">
                        <Bot size={18} />
                      </div>
                    ) : (
                      <div className="w-full h-full bg-zinc-700 border-zinc-600 flex items-center justify-center rounded-full text-zinc-200">
                        <UserIcon size={18} />
                      </div>
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm text-zinc-400 mb-1 flex items-center">
                      {msg.role === "assistant" ? "Adaptive RAG" : "You"}
                    </div>
                    <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800 max-w-none text-zinc-300 text-[15px]">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>

                    {msg.role === "assistant" && msg.references && msg.references.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-zinc-800/50 flex flex-wrap gap-2">
                        {msg.references.map((ref, rIdx) => (
                          <button
                            key={rIdx}
                            onClick={() => openPdf(ref.source, ref.page, ref.snippet)}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-md text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors shadow-sm"
                            title={ref.snippet}
                          >
                            <FileText size={14} className="text-indigo-400" />
                            {ref.source.split(/[/\\]/).pop()} <span className="text-zinc-500 ml-1">p.{ref.page}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div className="flex flex-col max-w-4xl mx-auto">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-zinc-800 border-zinc-700 shadow-sm text-indigo-400">
                  <Bot size={18} />
                </div>
                <div className="flex-1 min-w-0 mt-1.5">
                  <div className="flex items-center gap-1.5 w-max px-4 py-2.5 bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm text-sm text-zinc-400">
                    <div className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce" style={{ animationDelay: "0ms" }}></div>
                    <div className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce" style={{ animationDelay: "150ms" }}></div>
                    <div className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce" style={{ animationDelay: "300ms" }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="h-32 md:h-40 flex-shrink-0 w-full" />
          <div ref={messagesEndRef} />
        </div>

        {/* Action Input */}
        <div className="absolute flex flex-col bottom-0 left-0 right-0 bg-gradient-to-t from-zinc-950 via-zinc-950 to-transparent pt-10 pb-6 px-4 md:px-8">
          <div className="max-w-4xl mx-auto w-full relative group">
            <form onSubmit={sendMessage} className="relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage(e);
                  }
                }}
                placeholder="Message Adaptive RAG..."
                className="w-full bg-zinc-900 border border-zinc-700/50 text-zinc-100 rounded-2xl px-5 py-4 pr-16 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500/50 resize-none overflow-hidden max-h-32 text-base shadow-sm transition-shadow disabled:opacity-50"
                rows={1}
                disabled={isLoading}
                style={{ minHeight: "60px" }}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className="absolute right-3 bottom-3 p-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white transition-colors"
              >
                <ChevronRight size={20} strokeWidth={2.5} />
              </button>
            </form>
            <div className="text-center mt-3 text-[11px] text-zinc-500 font-medium">
              Adaptive RAG can make mistakes. Verify important information.
            </div>
          </div>
        </div>
      </div>

      {/* Right Document Viewer Panel */}
      {viewingPdf && (
        <div className="w-1/2 flex-1 flex flex-col bg-zinc-900 shadow-2xl relative transition-all duration-300 border-l border-zinc-800 z-20">
          <div className="h-14 flex items-center justify-between px-4 border-b border-zinc-800 bg-zinc-950/50 flex-shrink-0">
            <div className="flex items-center gap-2 text-sm text-zinc-300 font-medium truncate pr-4">
              <FileText size={16} className="text-indigo-400" />
              <span className="truncate">{viewingPdf.split('/').pop()?.split('#')[0]}</span>
            </div>
            <button
              onClick={() => setViewingPdf(null)}
              className="p-1.5 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 rounded-md transition-colors"
            >
              <X size={18} />
            </button>
          </div>
          <div className="flex-1 bg-[#525659] relative w-full h-full">
            <iframe
              src={viewingPdf}
              className="w-full h-full border-none absolute inset-0"
              title="PDF Viewer"
            ></iframe>
          </div>
        </div>
      )}
    </div>
  );
}
