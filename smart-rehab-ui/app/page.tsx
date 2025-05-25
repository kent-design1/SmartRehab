// app/route.ts
import Link from "next/link";
import { HeroBg } from "@/assets";

export default function Home() {
    return (
        <section
            className="relative h-screen w-full flex flex-col items-center justify-center text-center overflow-hidden bg-cover bg-center"
            style={{ backgroundImage: `url(${HeroBg.src})` }}
        >
            {/* Overlay */}
            <div className="absolute inset-0 bg-black/40" />

            {/* Hero Content */}
            <div className="relative z-10 max-w-3xl px-6 space-y-6">
                <h1
                    className="
            bg-gradient-to-r from-white via-blue-100 to-blue-200
            bg-clip-text text-transparent
            font-extrabold
            uppercase
            tracking-tight
            leading-tight
            text-4xl sm:text-5xl md:text-6xl lg:text-7xl
            drop-shadow-lg
          "
                >
                    SmartRehab SCIM Predictor
                </h1>

          {/*      <p*/}
          {/*          className="*/}
          {/*  mx-auto*/}
          {/*  text-gray-200/90*/}
          {/*  italic*/}
          {/*  max-w-prose*/}
          {/*  text-lg sm:text-xl md:text-2xl*/}
          {/*  tracking-wide*/}
          {/*  drop-shadow*/}
          {/*"*/}
          {/*      >*/}
          {/*          Instantly forecast patient recovery scores at{" "}*/}
          {/*          <span className="font-semibold text-white">6</span>,{" "}*/}
          {/*          <span className="font-semibold text-white">12</span>,{" "}*/}
          {/*          <span className="font-semibold text-white">18</span> &{" "}*/}
          {/*          <span className="font-semibold text-white">24</span> weeks.*/}
          {/*      </p>*/}

                <Link
                    href="/predict"
                    className="
            inline-block
            bg-blue-500 hover:bg-blue-600
            text-white font-medium
            py-3 px-10
            rounded-full
            shadow-xl
            transition
            transform hover:-translate-y-1
          "
                >
                    Get Started
                </Link>
            </div>
        </section>
    );
}