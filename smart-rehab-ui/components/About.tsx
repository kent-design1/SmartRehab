// app/about/route.ts
import Image from 'next/image';
import { HeroBg } from '@/assets';


const team = [
    { name: 'Dr. Alice Smith', role: 'Lead Data Scientist', contribution: 'Model design & feature engineering' },
    { name: 'Dr. Brian Lee', role: 'Clinical Advisor', contribution: 'Clinical validation & domain expertise' },
    { name: 'Clara Zhao', role: 'Frontend Engineer', contribution: 'UI/UX and component development' },
    { name: 'David Müller', role: 'Backend Engineer', contribution: 'API development & integration' },
];

export default function AboutPage() {
    return (
        <main className=" w-full mt-4">
            {/* Hero Image */}
            <section className="w-full h-64 md:h-[30rem] relative">
                <Image
                    src={HeroBg}
                    alt="About SmartRehab Hero"
                    fill
                    className="object-cover w-full"
                />
                <div className="absolute inset-0 bg-black/30"/>
                <div className="absolute inset-0 flex items-center justify-center">
                    <h1 className="text-3xl md:text-5xl lg:text-6xl font-extrabold text-white drop-shadow-lg">
                        About SmartRehab
                    </h1>
                </div>
            </section>

            <section className="container mx-auto px-6 lg:px-24 py-12 text-gray-700 w-full">
                <p className="text-lg md:text-xl leading-relaxed mx-auto mb-6">
                    In today’s clinical landscape, understanding individual patient trajectories is paramount.
                    SmartRehab harnesses an extensive, Swiss-realistic rehabilitation dataset—featuring demographics,
                    laboratory results, psychosocial assessments, and early functional scores—to train advanced ensemble
                    models. By combining Extra Trees and Gradient Boosting with domain-driven feature engineering, our
                    platform captures nuanced recovery patterns, accommodates patient variability, and anticipates
                    potential plateaus or rapid improvements.
                </p>
                <p className="text-lg md:text-xl leading-relaxed mx-auto mb-6">
                    At each key milestone (weeks 6, 12, 18, and 24), clinicians receive predictive SCIM scores, complete
                    with confidence intervals and interpretable feature contributions. This transparency ensures that
                    healthcare teams can trust algorithmic insights when adjusting therapy intensity, medication plans,
                    or discharge timing.
                </p>
                <p className="text-lg md:text-xl leading-relaxed mx-auto">
                    Beyond personalized treatment, SmartRehab optimizes resource management by identifying patients at
                    risk of extended stays or needing additional interventions, ultimately driving efficiency and
                    elevating patient care standards across rehabilitation centers.
                </p>
            </section>

            {/* Team Bento Grid */}
            <section className="container mx-auto px-6 lg:px-24 pb-16">
                <h2 className="text-2xl md:text-3xl font-semibold text-gray-900 mb-8 text-center">
                    Meet the Team
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
                    {team.map((member, idx) => (
                        <div
                            key={idx}
                            className="bg-white rounded-2xl shadow-lg overflow-hidden hover:shadow-2xl transition-shadow duration-300"
                        >
                            <div className="relative h-40">
                                <Image
                                    src={HeroBg}
                                    alt={member.name}
                                    fill
                                    className="object-cover"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/50"/>
                                <div className="absolute bottom-2 left-2 text-white">
                                    <p className="font-bold text-lg drop-shadow">{member.name}</p>
                                    <p className="text-sm drop-shadow">{member.role}</p>
                                </div>
                            </div>
                            <div className="p-4">
                                <p className="text-gray-600 text-sm leading-snug">
                                    {member.contribution}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </section>
        </main>
    );
}