import { ReactNode } from "react";

export default function PredictLayout({ children }: { children: ReactNode }) {
    return <div className="py-4">{children}</div>;
}