"use client"

import { useState } from "react";

export default function SampleInputAndResponse ({ className }) {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    let input = e.target.input.value;

    setLoading(true);
    
    // make api call
    const response = await fetch(process.env.NEXT_PUBLIC_RL_API_ENDPOINT, {
      method: "POST",
      body: input,
      headers: {
        "Content-Type": "text/plain"
      }
    });

    const output = await response.text();
    setResponse(output);
    setLoading(false);
  }

  // Define button states
  const activeButtonClass = "bg-green-500 hover:bg-green-600 active:bg-green-700";
  const inactiveButtonClass = "bg-gray-300 cursor-not-allowed";

  return (
    <>
      <div className="font-bold mt-5 text-lg">Reinforcement Learning</div>
      <div className={className}>
        <form onSubmit={handleSubmit} className="flex flex-col items-start">
          <textarea name="input" placeholder="Input Stakeholder Features (CSV format)" required className="border-2 border-black w-1/2" />
          <button type="submit" className={loading ? inactiveButtonClass : activeButtonClass} disabled={loading}>Submit</button>
        </form>

        <div className="font-bold">Response</div>
        {loading ?
          <div className="text-gray-500">Running RL...</div>
          :
          (response ?
            <div className="whitespace-pre-wrap">{response}</div>
            :
            <div className="text-gray-500">No RL response yet</div>
          )
        }
      </div>
    </>
  )
}