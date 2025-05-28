// contact/route
'use client';
import { useState } from 'react';
import { Mail, Phone, MapPin, User } from 'lucide-react';
import Image from 'next/image';
import { HeroBg } from '@/assets';

export default function ContactPage() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [subject, setSubject] = useState('');
    const [message, setMessage] = useState('');
    const [status, setStatus] = useState<'idle' | 'sending' | 'success' | 'error'>('idle');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setStatus('sending');
        try {
            const res = await fetch('/api/contact', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, subject, message }),
            });
            if (res.ok) setStatus('success'); else throw new Error();
        } catch {
            setStatus('error');
        }
    };

    return (
        <main className="bg-gradient-to-br from-blue-50 to-white min-h-screen mb-24 ">
            {/* Hero Banner */}
            <section className="relative h-64 md:h-96 mb-12">
                <Image src={HeroBg} fill alt="Contact Banner" className="object-cover" />
                <div className="absolute inset-0 bg-black/40" />
                <div className="absolute inset-0 flex items-center justify-center">
                    <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold text-white drop-shadow-lg">
                        Contact Us
                    </h1>
                </div>
            </section>

            <section className="container mx-auto px-6 lg:px-24 flex flex-row w-full gap-8">
                {/* Contact Info Card */}
                <div className="  flex flex-col  justify-center items-start max-w-lg">
                    <h2 className="text-2xl font-bold text-gray-900">Contact Information</h2>
                    <p className="text-gray-700 leading-relaxed">
                        Have questions or feedback? Reach out to our team using the details below or send us a message.
                    </p>
                    <div className="space-y-4">
                        <div className="flex items-center space-x-3">
                            <MapPin className="w-6 h-6 text-blue-600" />
                            <span className="text-gray-800">Zurich, Switzerland</span>
                        </div>
                        <div className="flex items-center space-x-3">
                            <Phone className="w-6 h-6 text-blue-600" />
                            <a href="tel:+41271234567" className="text-gray-800 hover:text-blue-600 transition">
                                +41 27 123 4567
                            </a>
                        </div>
                        <div className="flex items-center space-x-3">
                            <Mail className="w-6 h-6 text-blue-600" />
                            <a href="mailto:contact@smartrehab.com" className="text-gray-800 hover:text-blue-600 transition">
                                contact@smartrehab.com
                            </a>
                        </div>
                    </div>
                </div>

                {/* Spacer for alignment on large screens */}
                <div className="hidden lg:block"></div>

                <div className="flex items-center flex-2">
                {/* Contact Form Card */}
                <form
                    onSubmit={handleSubmit}
                    className="bg-white rounded-3xl shadow-lg p-8 space-y-6 transform transition hover:shadow-2xl hover:-translate-y-1 lg:col-span-2"
                >
                    <h2 className="text-2xl font-bold text-gray-900">Send Us a Message</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Name Input */}
                        <div className="relative">
                            <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Your Name"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                required
                                className="pl-10 w-full border border-gray-200 rounded-lg p-3 focus:ring-blue-500 focus:border-blue-500 transition"
                            />
                        </div>

                        {/* Email Input */}
                        <div className="relative">
                            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                            <input
                                type="email"
                                placeholder="Your Email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                className="pl-10 w-full border border-gray-200 rounded-lg p-3 focus:ring-blue-500 focus:border-blue-500 transition"
                            />
                        </div>
                    </div>

                    {/* Subject */}
                    <div className="relative">
                        <input
                            type="text"
                            placeholder="Subject"
                            value={subject}
                            onChange={(e) => setSubject(e.target.value)}
                            required
                            className="w-full border border-gray-200 rounded-lg p-3 focus:ring-blue-500 focus:border-blue-500 transition"
                        />
                    </div>

                    {/* Message */}
                    <textarea
                        placeholder="Your Message"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        required
                        className="w-full border border-gray-200 rounded-lg p-3 h-40 resize-none focus:ring-blue-500 focus:border-blue-500 transition"
                    />

                    {/* Submit */}
                    <button
                        type="submit"
                        disabled={status === 'sending'}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg shadow-md transition transform hover:-translate-y-0.5 disabled:opacity-50"
                    >
                        {status === 'sending' ? 'Sending...' : 'Send Message'}
                    </button>
                    {status === 'success' && (
                        <p className="text-green-600 font-medium text-center">Thank you! We'll get back to you soon.</p>
                    )}
                    {status === 'error' && (
                        <p className="text-red-600 font-medium text-center">Oops! Something went wrong.</p>
                    )}
                </form>
                </div>
            </section>
        </main>
    );
}
