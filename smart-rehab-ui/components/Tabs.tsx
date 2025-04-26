interface TabsProps {
    weeks: number[];
    selected: number;
    onSelect: (wk: number) => void;
}

export default function Tabs({ weeks, selected, onSelect }: TabsProps) {
    return (
        <div className="flex space-x-4 border-b">
            {weeks.map((wk) => (
                <button
                    key={wk}
                    onClick={() => onSelect(wk)}
                    className={`pb-2 ${
                        selected === wk
                            ? "border-b-2 border-blue-600 text-blue-600"
                            : "border-transparent text-gray-600"
                    } hover:text-blue-500`}
                >
                    Week {wk}
                </button>
            ))}
        </div>
    );
}