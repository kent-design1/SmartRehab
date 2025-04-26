interface TabsProps {
    weeks: number[];
    selected: number;
    onSelect: (wk: number) => void;
}

export default function Tabs({ weeks, selected, onSelect }: TabsProps) {
    return (
        <div className="flex overflow-x-auto space-x-4 py-2 px-4  mx-auto justify-between items-center">
            {weeks.map((wk) => (
                <button
                    key={wk}
                    onClick={() => onSelect(wk)}
                    className={`relative whitespace-nowrap px-6 py-3 rounded-full font-semibold text-sm transition-all duration-300 ease-in-out focus:outline-none
                        ${selected === wk
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                        : 'bg-gray-100 text-gray-700 hover:bg-blue-50 hover:text-blue-600'}
                    `}
                >
                    Week {wk}
                    {selected === wk && (
                        <span className="absolute bottom-0 left-1/2 w-6 h-1 bg-white/70 rounded-full transform -translate-x-1/2 -translate-y-1" />
                    )}
                </button>
            ))}
        </div>
    );
}