"use client"

import { useState } from "react";

import pdfToText from "react-pdftotext";
import LLMInfoModal from "@/components/LLMInfoModal";

import { EXTRACTION_PROMPT_CORE } from "@/components/LLMInfoModal";
import { ActiveButton, InactiveButton, TealButtonSmall, LightGrayButton } from "@/components/Classes";

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
      const fileText = await pdfToText(file);
      textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${fileText}`;
    } else {
      let input = e.target.input.value;
      textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${input}`;
    }
    callGPT(textInput);
  }

  return (
    <>
      <div className="flex gap-3">
        <div className="font-bold text-lg">Stakeholder Extraction</div>
        <LLMInfoModal />
      </div>
      <div className='w-full'>
        {usePdf ?
          <form onSubmit={handleSubmit} className="flex flex-col items-start mt-3">
            <input type="file" name="file" accept=".pdf" required className="" />
            <div className="flex gap-3 mt-1">
              <button type="submit" className={loading ? InactiveButton : ActiveButton} disabled={loading}>Submit PDF</button>
              <button type="button" onClick={() => setUsePdf(false)} className={`${LightGrayButton}`}>Use text input</button>
            </div>
          </form>
          :
          <form onSubmit={handleSubmit} className="flex flex-col items-start w-full mt-3">
            <textarea name="input" rows="5" placeholder="Input prompt here" required className="border-2 border-black w-1/2" />
            <div className="flex gap-3 mt-1">
              <button type="submit" className={loading ? InactiveButton : ActiveButton} disabled={loading}>Submit</button>
              <button type="button" onClick={() => setUsePdf(true)} className={`${LightGrayButton}`}>Use PDF input (experimental)</button>
            </div>
          </form>
        }

        <div className="font-bold mt-3 text-lg">Response</div>
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