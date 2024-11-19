"use client"

import { useState } from "react";

import pdfToText from "react-pdftotext";

import { EXTRACTION_PROMPT_CORE } from "@/components/LLMInfoModal";

export default function TextStakeholderExtract () {
  const [usePdf, setUsePdf] = useState(false);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  const callGPT = async (textInput) => {
    setLoading(true);

    // make api call
    const response = await fetch("/api/gpt", {
      method: "POST",
      body: textInput
    });
    const output = await response.text();

    setResponse(output);
    setLoading(false);
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    let textInput = "";
    if (usePdf) {
      let file = e.target.file.files[0];
      pdfToText(file).then(async (text) => {
        textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${text}`;
      });
    } else {
      let input = e.target.input.value;
      textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${input}`;
    }
    callGPT(textInput);
  }

  // Define button states
  const activeButtonClass = "bg-green-500 hover:bg-green-600 active:bg-green-700";
  const inactiveButtonClass = "bg-gray-300 cursor-not-allowed";

  return (
    <>
      <div className="font-bold text-lg">Stakeholder Extraction</div>
      <div className='w-full'>
        {/* TODO: popup modal that lets you view definitions*/}
        {usePdf ?
            <button onClick={() => setUsePdf(false)} className="text-blue-500 underline">Use text input</button>
            :
            <button onClick={() => setUsePdf(true)} className="text-blue-500 underline">Use PDF input (experimental)</button>
        }

        {usePdf ?
          <form onSubmit={handleSubmit} className="flex flex-col items-start">
            <input type="file" name="file" accept=".pdf" required className="" />
            <button type="submit" className={loading ? inactiveButtonClass : activeButtonClass} disabled={loading}>Submit PDF</button>
          </form>
          :
          <form onSubmit={handleSubmit} className="flex flex-col items-start w-full">
            <textarea name="input" placeholder="Input prompt here" required className="border-2 border-black w-1/2" />
            <button type="submit" className={loading ? inactiveButtonClass : activeButtonClass} disabled={loading}>Submit</button>
          </form>
        }

        <div className="font-bold">Response</div>
        {loading ?
          <div className="text-gray-500">Loading...</div>
          :
          (response ?
            <div className="whitespace-pre-wrap">{response}</div>
            :
            <div className="text-gray-500">No LLM response yet</div>
          )
        }
      </div>
    </>
  )
}