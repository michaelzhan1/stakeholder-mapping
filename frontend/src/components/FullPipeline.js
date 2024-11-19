"use client"

import { useState } from "react";

import pdfToText from "react-pdftotext";

import { EXTRACTION_PROMPT_CORE } from "@/components/LLMInfoModal";

export default function FullPipeline ({ className }) {
  const [usePdf, setUsePdf] = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [rlLoading, setRlLoading] = useState(false);
  const [llmResponse, setLlmResponse] = useState(null);
  const [rlResponse, setRlResponse] = useState(null);

  const runRL = async (input) => {
    setRlLoading(true);

    const response = await fetch(process.env.NEXT_PUBLIC_RL_API_ENDPOINT, {
      method: "POST",
      body: input,
      headers: {
        "Content-Type": "text/plain"
      }
    });

    const output = await response.text();
    setRlResponse(output);

    setRlLoading(false);
  }

  const callGPT = async (textInput) => {
    setLlmLoading(true);

    // make api call for LLM extraction
    const response = await fetch("/api/gpt", {
      method: "POST",
      body: textInput
    });
    const output = await response.text();
    setLlmResponse(output);
    setLlmLoading(false);

    return output;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    let textInput = "";
    if (usePdf) {
      let file = e.target.file.files[0];
      const text = await pdfToText(file);
      textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${text}`;
    } else {
      let input = e.target.input.value;
      textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${input}`;
    }
    
    const gptOutput = await callGPT(textInput);
    runRL(gptOutput);
  }

  // Define button states
  const activeButtonClass = "bg-green-500 hover:bg-green-600 active:bg-green-700";
  const inactiveButtonClass = "bg-gray-300 cursor-not-allowed";

  return (
    <>
      <div className="font-bold text-lg">Stakeholder Extraction</div>
      <div className={className}>
        {/* TODO: popup modal that lets you view definitions*/}
        {usePdf ?
            <button onClick={() => setUsePdf(false)} className="text-blue-500 underline">Use text input</button>
            :
            <button onClick={() => setUsePdf(true)} className="text-blue-500 underline">Use PDF input (experimental)</button>
        }

        {usePdf ?
          <form onSubmit={handleSubmit} className="flex flex-col items-start">
            <input type="file" name="file" accept=".pdf" required className="" />
            <button type="submit" className={(rlLoading || llmLoading) ? inactiveButtonClass : activeButtonClass} disabled={rlLoading || llmLoading}>Submit PDF</button>
          </form>
          :
          <form onSubmit={handleSubmit} className="flex flex-col items-start">
            <textarea name="input" placeholder="Input prompt here" required className="border-2 border-black w-1/2" />
            <button type="submit" className={(rlLoading || llmLoading) ? inactiveButtonClass : activeButtonClass} disabled={rlLoading || llmLoading}>Submit</button>
          </form>
        }

        <div className="font-bold">LLM Response (limited to 5 stakeholders)</div>
        {llmLoading ?
          <div className="text-gray-500">Loading...</div>
          :
          (llmResponse ?
            <div className="whitespace-pre-wrap">{llmResponse}</div>
            :
            <div className="text-gray-500">No LLM response yet</div>
          )
        }
        
        <div className="font-bold">RL Response</div>
        {rlLoading ?
          <div className="text-gray-500">Running RL...</div>
          :
          (rlResponse ?
            <div className="whitespace-pre-wrap">{rlResponse}</div>
            :
            <div className="text-gray-500">No RL response yet</div>
          )
        }
      </div>
    </>
  )
}