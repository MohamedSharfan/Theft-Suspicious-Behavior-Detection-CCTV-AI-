"use client";

import { FormEvent, ReactNode, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Bot,
  BrainCircuit,
  CircleDot,
  Eye,
  LockKeyhole,
  MessageSquareText,
  Send,
  ShieldCheck,
  Siren,
  Sparkles,
  UserRound,
  Video
} from "lucide-react";

type SecurityEvent = {
  id: number;
  timestamp: string;
  person_id: number;
  risk_score: number;
  status: "normal" | "suspicious";
  reason: string;
  alert: string;
  camera: string;
  explanation: string;
  behavior_chain?: string[];
  recommendation?: string;
};

type ChatMessage = {
  id: number;
  role: "user" | "ai";
  text: string;
  recommendation?: string;
};


type Metrics = {
  subjects: number;
  suspicious: number;
  fps: number;
  camera_count: number;
};

const initialEvents: SecurityEvent[] = [];

const initialMessages: ChatMessage[] = [
  {
    id: 1,
    role: "ai",
    text: "Monitoring the live feed."
  }
];

function riskLabel(score: number) {
  if (score >= 0.8) return "High";
  if (score >= 0.6) return "Medium";
  return "Low";
}

function cleanText(value: unknown) {
  if (typeof value !== "string") {
    return "";
  }
  return value.replace(/\*\*/g, "").replace(/\n{3,}/g, "\n\n").trim();
}

function answerQuestion(question: string, latestEvent: SecurityEvent | null) {
  if (!latestEvent) {
    return "No suspicious activity detected yet.";
  }
  const q = question.toLowerCase();

  if (q.includes("show last suspicious event") || q.includes("last suspicious event") || q.includes("incident report")) {
    return `INCIDENT REPORT\n-----------------------\n\nTime: ${latestEvent.timestamp}\nCamera: ${latestEvent.camera}\nPerson ID: ${latestEvent.person_id}\nThreat Level: ${latestEvent.status.toUpperCase()}\nRisk Score: ${Math.round(
      latestEvent.risk_score * 100
    )}%\n\nReason:\n${latestEvent.reason}\n\nAI Explanation:\n${latestEvent.explanation}\n\nAlert Message:\n${latestEvent.alert}\n\nRecommendation:\n${latestEvent.recommendation ?? ""}`.trim();
  }

  if (q.includes("why") || q.includes("flagged")) {
    return `Person ${latestEvent.person_id} was flagged because the model detected ${latestEvent.reason}. Confidence is ${Math.round(
      latestEvent.risk_score * 100
    )}%, with status: ${latestEvent.status.toUpperCase()}.`;
  }

  if (q.includes("last") || q.includes("event")) {
    return `Last event: ${latestEvent.alert} at ${latestEvent.timestamp}. Explanation: ${latestEvent.explanation}`;
  }

  if (q.includes("risk") || q.includes("level")) {
    return `Current threat level is ${riskLabel(latestEvent.risk_score).toUpperCase()}. Highest active subject is Person ${
      latestEvent.person_id
    } with ${Math.round(latestEvent.risk_score * 100)}% risk.`;
  }

  return `Latest feed intelligence: Person ${latestEvent.person_id}, ${Math.round(
    latestEvent.risk_score * 100
  )}% risk, reason: ${latestEvent.reason}.`;
}

export default function DashboardPage() {
  const [events, setEvents] = useState<SecurityEvent[]>(initialEvents);
  const [metrics, setMetrics] = useState<Metrics>({
    subjects: 0,
    suspicious: 0,
    fps: 0,
    camera_count: 1
  });
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

  const latestEvent = events[0] ?? null;
  const isSuspicious = latestEvent?.status === "suspicious";
  const threatLevel = latestEvent ? riskLabel(latestEvent.risk_score) : "Low";

  useEffect(() => {
    let isActive = true;

    async function loadEvents() {
      try {
        const response = await fetch(`${apiBase}/api/events`, { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        if (!isActive) return;

        const incoming = Array.isArray(payload?.events) ? payload.events : [];
        const suspiciousIncoming = incoming.filter((event) => event.status === "suspicious");
        if (suspiciousIncoming.length > 0) {
          setEvents(suspiciousIncoming);
        } else if (payload?.latest_event?.status === "suspicious") {
          setEvents([payload.latest_event]);
        } else {
          setEvents([]);
        }

        if (payload?.metrics) {
          setMetrics(payload.metrics);
        }
      } catch (error) {
        console.warn("API unavailable", error);
      }
    }

    loadEvents();
    const timer = window.setInterval(loadEvents, 4000);
    return () => {
      isActive = false;
      window.clearInterval(timer);
    };
  }, [apiBase]);

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = input.trim();
    if (!question) return;

    setMessages((current) => [...current, { id: Date.now(), role: "user", text: question }]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      const payload = response.ok ? await response.json() : null;
      const answer = payload?.answer ?? answerQuestion(question, latestEvent);
      const recommendation = payload?.recommendation;

      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 2,
          role: "ai",
          text: answer,
          recommendation
        }
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 2,
          role: "ai",
          text: answerQuestion(question, latestEvent)
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  }

  return (
    <main className="control-grid relative min-h-screen overflow-hidden bg-[#020617] px-4 py-4 text-slate-100 sm:px-6 lg:px-8">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),transparent_16%,transparent_84%,rgba(39,232,255,0.05))]" />
      <div className="relative mx-auto flex max-w-[2000px] flex-col gap-4">
        <TopStatusBar threatLevel={threatLevel} isSuspicious={isSuspicious} metrics={metrics} />

        <section className="grid h-auto grid-cols-1 gap-4 xl:h-[calc(100vh-116px)] xl:grid-cols-[1.8fr_0.7fr_0.7fr]">
          <CameraPanel
            apiBase={apiBase}
            latestEvent={latestEvent}
            isSuspicious={isSuspicious}
          />
          <AnalysisPanel events={events} />
          <AssistantPanel
            messages={messages}
            input={input}
            isTyping={isTyping}
            onInput={setInput}
            onSubmit={submitQuestion}
          />
        </section>
      </div>

    </main>
  );
}

function TopStatusBar({
  threatLevel,
  isSuspicious,
  metrics
}: {
  threatLevel: string;
  isSuspicious: boolean;
  metrics: Metrics;
}) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel flex flex-col gap-3 rounded-lg px-4 py-3 lg:flex-row lg:items-center lg:justify-between"
    >
      <div className="flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-md border border-cyan-300/40 bg-cyan-300/10 shadow-cyan">
          <ShieldCheck className="h-5 w-5 text-cyan-200" />
        </div>
        <div>
          <h1 className="text-xl font-semibold tracking-[0.12em] text-white sm:text-2xl">CCTV AI MONITOR</h1>
          <p className="font-mono text-sm uppercase tracking-[0.16em] text-cyan-200/80">Single live feed / AI risk review</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3 lg:grid-cols-6 lg:min-w-[720px]">
        <StatusPill icon={<Activity />} label="System" value="Active" tone="cyan" />
        <StatusPill icon={<LockKeyhole />} label="Threat Level" value={threatLevel} tone={isSuspicious ? "amber" : "cyan"} />
        <StatusPill icon={<BrainCircuit />} label="AI Analysis" value="Live" tone="blue" />
        <StatusPill icon={<UserRound />} label="Subjects" value={`${metrics.subjects}`} tone="blue" />
        <StatusPill icon={<AlertTriangle />} label="Suspicious" value={`${metrics.suspicious}`} tone={metrics.suspicious > 0 ? "amber" : "cyan"} />
        <StatusPill icon={<Video />} label="AI FPS" value={`${metrics.fps}`} tone="cyan" />
      </div>
    </motion.header>
  );
}

function StatusPill({
  icon,
  label,
  value,
  tone
}: {
  icon: ReactNode;
  label: string;
  value: string;
  tone: "cyan" | "blue" | "amber";
}) {
  const color =
    tone === "amber"
      ? "text-amber-200 border-amber-300/30 bg-amber-400/10"
      : tone === "blue"
        ? "text-blue-200 border-blue-300/25 bg-blue-400/10"
        : "text-cyan-200 border-cyan-300/25 bg-cyan-400/10";

  return (
    <div className={`flex items-center gap-2 rounded-md border px-3 py-2 ${color}`}>
      <span className="h-4 w-4 [&>svg]:h-4 [&>svg]:w-4">{icon}</span>
      <div>
        <p className="text-[11px] uppercase tracking-[0.18em] opacity-70">{label}</p>
        <p className="font-mono text-base font-semibold">{value}</p>
      </div>
    </div>
  );
}

function CameraPanel({
  apiBase,
  latestEvent,
  isSuspicious
}: {
  apiBase: string;
  latestEvent: SecurityEvent | null;
  isSuspicious: boolean;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, x: -18 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-panel flex min-h-[620px] flex-col rounded-lg p-5 xl:min-h-0 xl:overflow-hidden"
    >
      <PanelTitle icon={<Video />} title="Live CCTV Feed" subtitle="AI anomaly model" />

      <motion.div
        animate={isSuspicious ? { scale: [1, 1.002, 1] } : { scale: 1 }}
        transition={{ duration: 1.2, repeat: isSuspicious ? Infinity : 0 }}
        className={`camera-noise relative mt-5 aspect-video w-full flex-1 overflow-hidden rounded-lg border bg-[#070d18] ${
          isSuspicious
            ? "border-amber-300/70 shadow-[0_0_24px_rgba(251,191,36,0.28)]"
            : "border-cyan-300/25 shadow-cyan"
        }`}
      >
        <img
          src={`${apiBase}/api/stream`}
          alt="Live CCTV stream"
          className="absolute inset-0 h-full w-full object-contain opacity-95"
          loading="eager"
        />
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_24%,rgba(79,140,255,0.24),transparent_24rem),linear-gradient(145deg,#07111f,#101827_44%,#070b14)] opacity-40" />
        <div className="absolute inset-x-0 top-0 h-16 bg-gradient-to-b from-cyan-300/10 to-transparent" />

        <div className="absolute left-5 top-5 flex items-center gap-2 rounded-md border border-cyan-300/25 bg-black/40 px-3 py-2 font-mono text-xs text-cyan-100 backdrop-blur">
          <CircleDot className="h-3.5 w-3.5 animate-pulse text-red-300" />
          LIVE FEED
        </div>


        {isSuspicious && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute right-5 top-5 flex items-center gap-2 rounded-md border border-amber-300/50 bg-amber-400/15 px-3 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-amber-100 backdrop-blur"
          >
            <AlertTriangle className="h-4 w-4" />
            Alert
          </motion.div>
        )}


        {isSuspicious && latestEvent && (
          <div className="absolute bottom-4 left-4 right-4 grid gap-3 md:grid-cols-3">
            <CameraMetric label="Person ID" value={`${latestEvent.person_id}`} danger={isSuspicious} />
            <CameraMetric label="Risk Score" value={`${Math.round(latestEvent.risk_score * 100)}%`} danger={isSuspicious} />
            <CameraMetric label="Status" value={latestEvent.status} danger={isSuspicious} />
          </div>
        )}
      </motion.div>

      <div className="mt-5 grid gap-3 md:grid-cols-3">
        {isSuspicious && latestEvent && (
          <>
            <SummaryTile label="Tracked subject" value={`Person ${latestEvent.person_id}`} danger={isSuspicious} />
            <SummaryTile label="Behavior signal" value={latestEvent.reason} danger={isSuspicious} />
            <SummaryTile label="AI status" value={latestEvent.status} danger={isSuspicious} />
          </>
        )}
      </div>
    </motion.section>
  );
}


function CameraMetric({ label, value, danger = false }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className={`rounded-md border px-3 py-2 backdrop-blur ${danger ? "border-amber-300/40 bg-amber-400/10 text-amber-100" : "border-cyan-300/20 bg-black/30 text-cyan-100"}`}>
      <p className="font-mono text-[11px] uppercase tracking-[0.2em] opacity-70">{label}</p>
      <p className="mt-1 text-xl font-semibold capitalize">{value}</p>
    </div>
  );
}

function SummaryTile({ label, value, danger = false }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className={`rounded-md border px-3 py-3 ${danger ? "border-amber-300/30 bg-amber-400/10 text-amber-100" : "border-slate-600/50 bg-white/5 text-slate-100"}`}>
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-slate-400">{label}</p>
      <p className="mt-1 line-clamp-1 text-base font-medium capitalize">{value}</p>
    </div>
  );
}

function AnalysisPanel({ events }: { events: SecurityEvent[] }) {
  const suspiciousEvents = events.filter((event) => event.status === "suspicious");
  const genaiEvent = suspiciousEvents[0] ?? null;
  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.08 }}
      className="glass-panel flex min-h-[620px] flex-col rounded-lg p-5 xl:min-h-0 xl:overflow-hidden"
    >
      <PanelTitle icon={<BrainCircuit />} title="AI Security Analysis" subtitle="Explainable anomaly intelligence" />

      {suspiciousEvents.length > 0 && (
        <>
          <div className="mt-4 rounded-lg border border-amber-300/30 bg-amber-400/10 p-4 shadow-[0_18px_50px_rgba(251,191,36,0.12)]">
            <div className="flex items-center gap-2 text-amber-100">
              <Siren className="h-5 w-5" />
              <p className="text-base font-semibold uppercase tracking-[0.2em]">Real-time Alerts Feed</p>
            </div>
            <div className="mt-3 flex items-center justify-between gap-3">
              <p className="text-sm text-amber-100/70">Report ready for latest alert.</p>
              <button
                type="button"
                onClick={() => window.open(`${apiBase}/api/report/latest`, "_blank")}
                className="rounded-md border border-amber-300/40 bg-amber-400/15 px-3 py-2 text-sm font-semibold uppercase tracking-[0.18em] text-amber-100 transition hover:bg-amber-400/25"
              >
                Download report (PDF)
              </button>
            </div>
            <div className="mt-4 max-h-56 space-y-3 overflow-y-auto pr-1">
              <AnimatePresence initial={false}>
                {suspiciousEvents.map((event) => (
                  <motion.article
                    key={event.id}
                    initial={{ opacity: 0, x: 24 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -24 }}
                    className="rounded-md border border-amber-300/40 bg-amber-400/10 p-3"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="font-mono text-sm text-slate-300">{event.timestamp}</span>
                      <span className="rounded-sm bg-amber-300/20 px-2 py-1 font-mono text-[11px] uppercase text-amber-100">
                        Person {event.person_id}
                      </span>
                    </div>
                    <p className="mt-2 whitespace-pre-line text-base text-slate-100">{cleanText(event.explanation)}</p>
                  </motion.article>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {genaiEvent && (
            <motion.div
              key={genaiEvent.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -14 }}
              className="mt-4 rounded-lg border border-amber-300/35 bg-amber-400/10 p-4 shadow-[0_18px_50px_rgba(251,191,36,0.14)]"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-amber-200" />
                  <p className="text-base font-semibold uppercase tracking-[0.2em]">AI Explanation</p>
                </div>
                <span className="font-mono text-sm text-slate-300">{genaiEvent.timestamp}</span>
              </div>
              <p className="mt-4 whitespace-pre-line text-xl leading-8 text-white">{cleanText(genaiEvent.explanation)}</p>
              <div className="mt-4 grid grid-cols-3 gap-2">
                <InsightMetric label="Risk" value={`${Math.round(genaiEvent.risk_score * 100)}%`} />
                <InsightMetric label="Signal" value={genaiEvent.reason} />
                <InsightMetric label="Alert" value={genaiEvent.alert} />
              </div>
              {Array.isArray(genaiEvent.behavior_chain) && genaiEvent.behavior_chain.length > 0 && (
                <div className="mt-4 space-y-2">
                  {genaiEvent.behavior_chain.map((step, index) => (
                    <div key={`${genaiEvent.id}-step-${index}`} className="rounded-md border border-white/10 bg-black/20 px-3 py-2 text-base text-slate-200">
                      {index + 1}. {step}
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </>
      )}

      {suspiciousEvents.length > 0 && (
        <div className="mt-4 rounded-lg border border-amber-300/20 bg-black/20 p-4">
          <p className="text-sm uppercase tracking-[0.18em] text-amber-200">Activity Timeline</p>
          <div className="mt-4 space-y-3">
            {suspiciousEvents.slice(0, 5).map((event) => (
              <div
                key={`timeline-${event.id}`}
                className="flex items-center justify-between rounded-md border border-white/10 bg-white/5 px-3 py-2"
              >
                <div>
                  <p className="text-base text-white">Person {event.person_id}</p>
                  <p className="text-sm text-slate-400">{event.reason}</p>
                </div>
                <span className="text-sm font-semibold text-amber-200">
                  {Math.round(event.risk_score * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.section>
  );
}

function InsightMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-black/20 p-2">
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-slate-300">{label}</p>
      <p className="mt-1 line-clamp-2 text-sm text-slate-100">{value}</p>
    </div>
  );
}

function AssistantPanel({
  messages,
  input,
  isTyping,
  onInput,
  onSubmit
}: {
  messages: ChatMessage[];
  input: string;
  isTyping: boolean;
  onInput: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const prompts = [
    "Why was Person 2 flagged?",
    "Show last suspicious event",
    "What is current risk level?",
    "List suspicious events from last 10 minutes",
    "What is the latest alert message?",
    "Which camera is active?"
  ];

  return (
    <motion.section
      initial={{ opacity: 0, x: 18 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.12 }}
      className="glass-panel flex min-h-[620px] flex-col rounded-lg p-5 xl:min-h-0 xl:overflow-hidden"
    >
      <PanelTitle icon={<MessageSquareText />} title="CCTV AI Security Assistant" subtitle="Conversational threat analysis" />

      <div className="mt-5 flex gap-2 overflow-x-auto pb-1">
        {prompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onInput(prompt)}
            className="whitespace-nowrap rounded-md border border-cyan-300/20 bg-cyan-300/10 px-3 py-2 text-sm text-cyan-100 transition hover:border-cyan-200/60 hover:bg-cyan-300/20"
          >
            {prompt}
          </button>
        ))}
      </div>

      <div className="mt-4 h-[360px] min-h-0 flex-1 space-y-4 overflow-y-auto rounded-lg border border-slate-700/70 bg-black/20 p-3 xl:h-auto xl:max-h-none">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-2 ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {message.role === "ai" && (
                <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-cyan-300/30 bg-cyan-300/10">
                  <Bot className="h-4 w-4 text-cyan-100" />
                </div>
              )}
              <div
                className={`max-w-[84%] whitespace-pre-line rounded-lg border px-3 py-2 text-base leading-7 ${
                  message.role === "user"
                    ? "border-blue-300/30 bg-blue-500/15 text-blue-50"
                    : "border-cyan-300/25 bg-cyan-400/10 text-slate-100 shadow-cyan"
                }`}
              >
                {cleanText(message.text)}
                {message.role === "ai" && message.recommendation && (
                  <div className="mt-2 rounded-md border border-red-300/20 bg-red-500/10 p-2 text-sm text-red-100">
                    Recommendation: {message.recommendation}
                  </div>
                )}
              </div>
              {message.role === "user" && (
                <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-blue-300/30 bg-blue-300/10">
                  <UserRound className="h-4 w-4 text-blue-100" />
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {isTyping && (
          <div className="flex items-center gap-2">
            <Bot className="h-4 w-4 text-cyan-200" />
            <div className="flex gap-1 rounded-md border border-cyan-300/20 bg-cyan-300/10 px-3 py-2">
              <span className="typing-dot h-2 w-2 rounded-full bg-cyan-200" />
              <span className="typing-dot h-2 w-2 rounded-full bg-cyan-200" />
              <span className="typing-dot h-2 w-2 rounded-full bg-cyan-200" />
            </div>
          </div>
        )}
      </div>

      <form onSubmit={onSubmit} className="mt-4 flex shrink-0 gap-2 rounded-lg border border-slate-700/70 bg-white/5 p-2">
        <input
          value={input}
          onChange={(event) => onInput(event.target.value)}
          placeholder="Ask the AI security assistant..."
          className="min-w-0 flex-1 rounded-md border border-cyan-300/20 bg-cyan-300/10 px-3 py-3 text-base text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-200/70 focus:shadow-cyan"
        />
        <button
          type="submit"
          className="flex h-12 w-12 shrink-0 items-center justify-center rounded-md border border-cyan-300/40 bg-cyan-300/15 text-cyan-100 transition hover:bg-cyan-300/25 hover:shadow-cyan"
          aria-label="Send question"
        >
          <Send className="h-5 w-5" />
        </button>
      </form>
    </motion.section>
  );
}

function PanelTitle({
  icon,
  title,
  subtitle,
  compact = false
}: {
  icon: ReactNode;
  title: string;
  subtitle: string;
  compact?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-3">
        <div className={`${compact ? "h-8 w-8" : "h-10 w-10"} flex items-center justify-center rounded-md border border-cyan-300/30 bg-cyan-300/10 text-cyan-100 shadow-cyan [&>svg]:h-5 [&>svg]:w-5`}>
          {icon}
        </div>
        <div>
          <h2 className={`${compact ? "text-base" : "text-lg"} font-semibold uppercase tracking-[0.18em] text-white`}>{title}</h2>
          <p className="font-mono text-[12px] uppercase tracking-[0.18em] text-slate-300">{subtitle}</p>
        </div>
      </div>
      {!compact && <Eye className="h-5 w-5 text-cyan-200/70" />}
    </div>
  );
}

