"use client"

import { useState } from "react";

import { EXTRACTION_PROMPT_CORE } from "@/components/LLMInfoModal";
import { LightGrayButton, ActiveButton, InactiveButton } from "@/components/Classes";

import pdfToText from "react-pdftotext";
import LLMInfoModal from "@/components/LLMInfoModal";


export default function FullPipeline () {
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
      textInput = EXTRACTION_PROMPT_CORE + "Limit to 5 stakeholders\n" + `Text data:\n${text}`;
    } else {
      let input = e.target.input.value;
      textInput = EXTRACTION_PROMPT_CORE + "Limit to 5 stakeholders\n" + `Text data:\n${input}`;
    }
    
    const gptOutput = await callGPT(textInput);
    runRL(gptOutput);
  }

  return (
    <>
      <div className="flex gap-3">
        <div className="font-bold text-lg">Stakeholder Extraction</div>
        <LLMInfoModal />
      </div>
      <div className='w-full'>
        {usePdf ?
          <form onSubmit={handleSubmit} className="flex flex-col items-start">
            <input type="file" name="file" accept=".pdf" required className="" />
            <div className="flex gap-3 mt-1">
              <button type="submit" className={(rlLoading || llmLoading) ? InactiveButton : ActiveButton} disabled={rlLoading || llmLoading}>Submit PDF</button>
              <button type="button" onClick={() => setUsePdf(false)} className={`${LightGrayButton}`}>Use text input</button>
            </div>
          </form>
          :
          <form onSubmit={handleSubmit} className="flex flex-col items-start mt-3">
            <textarea name="input" rows="5" placeholder="Input prompt here" required className="border-2 border-black w-1/2" />
            <div className="flex gap-3 mt-1">
              <button type="submit" className={(rlLoading || llmLoading) ? InactiveButton : ActiveButton} disabled={rlLoading || llmLoading}>Submit</button>
              <button type="button" onClick={() => setUsePdf(true)} className={`${LightGrayButton}`}>Use PDF input (experimental)</button>
            </div>
          </form>
        }

        <hr className='w-full my-3' />

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

        <hr className='w-full my-3' />
        
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