// app/docs/page.tsx
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';

export default function ApiDocsPage() {
    return (
        <main className="min-h-screen bg-gray-50 py-16 px-6 lg:px-24">
            <h1 className="text-4xl font-extrabold text-gray-900 mb-8">API Documentation</h1>
            <section className="space-y-12">
                {/* Login Endpoint */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h2 className="text-2xl font-semibold text-gray-800 mb-4">POST <code className="bg-gray-100 px-1 rounded">/api/login</code></h2>
                    <p className="text-gray-700 mb-4">
                        Authenticate with shared credentials to receive an HTTP-only cookie for subsequent requests.
                    </p>
                    <h3 className="font-medium text-gray-800">Request</h3>
                    <SyntaxHighlighter language="json" style={atomDark} className="rounded-lg">
                        {`{
  "email": "doctor@yourhospital.org",
  "password": "SuperSecretPass123"
}`}
                    </SyntaxHighlighter>
                    <h3 className="mt-4 font-medium text-gray-800">Response</h3>
                    <SyntaxHighlighter language="json" style={atomDark} className="rounded-lg">
                        {`// 200 OK
{
  "success": true
}`}
                    </SyntaxHighlighter>
                </div>

                {/* Logout Endpoint */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h2 className="text-2xl font-semibold text-gray-800 mb-4">POST <code className="bg-gray-100 px-1 rounded">/api/logout</code></h2>
                    <p className="text-gray-700 mb-4">
                        Clear authentication cookie to sign out the user.
                    </p>
                    <h3 className="font-medium text-gray-800">Response</h3>
                    <SyntaxHighlighter language="json" style={atomDark} className="rounded-lg">
                        {`// 200 OK
{
  "success": true
}`}
                    </SyntaxHighlighter>
                </div>

                {/* Predict Endpoint */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                    <h2 className="text-2xl font-semibold text-gray-800 mb-4">POST <code className="bg-gray-100 px-1 rounded">/api/predict/week{'{week}'}</code></h2>
                    <p className="text-gray-700 mb-4">
                        Predict SCIM score for a given week. Replace <code>{'{week}'}</code> with 6, 12, 18, or 24.
                    </p>
                    <h3 className="font-medium text-gray-800">Request Payload</h3>
                    <SyntaxHighlighter language="json" style={atomDark} className="rounded-lg">
                        {`{
  "features": {
    "Total_SCIM_0": 10,
    // Additional numeric or categorical features
    "Age": 65,
    "Gender": "Male",
    // Optional intermediate SCIM values...
    "Total_SCIM_6": 20
  }
}`}
                    </SyntaxHighlighter>
                    <h3 className="mt-4 font-medium text-gray-800">Response</h3>
                    <SyntaxHighlighter language="json" style={atomDark} className="rounded-lg">
                        {`// 200 OK
{
  "prediction": 50.79824480457508
}`}
                    </SyntaxHighlighter>
                </div>
            </section>
        </main>
    );
}
