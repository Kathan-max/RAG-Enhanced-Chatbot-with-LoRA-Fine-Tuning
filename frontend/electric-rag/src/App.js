import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, Upload, Database, Send, X, FileText, CheckCircle, Loader, Zap, Brain, Sparkles, ChevronDown, ChevronUp, Eye, EyeOff } from 'lucide-react';

const App = () => {
  const [activePanel, setActivePanel] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [trainModel, setTrainModel] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState({}); // Track which reasoning sections are expanded
  
  // LLM Configuration States
  const [useMultiLLM, setUseMultiLLM] = useState(false);
  const [activeLLMs, setActiveLLMs] = useState([]);
  const [expertLLM, setExpertLLM] = useState('');
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const availableLLMs = [
    'gpt-3',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-4o',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4-32k',
    'Claude 2',
    'Claude Instant 2',
    'Claude 3',
    'Gemini-Pro',
    'Llama-2',
    'Mistral-7B'
  ];

  const expertLLMOptions = [
    'gpt-3',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-4o',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4-32k',
    'Claude 2',
    'Claude Instant 2',
    'Claude 3',
    'Gemini-Pro',
    'Llama-2',
    'Mistral-7B'
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleReasoning = (messageIndex) => {
    setExpandedReasoning(prev => ({
      ...prev,
      [messageIndex]: !prev[messageIndex]
    }));
  };

  const handleLLMToggle = (llm) => {
    setActiveLLMs(prev => 
      prev.includes(llm) 
        ? prev.filter(l => l !== llm)
        : [...prev, llm]
    );
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const payload = {
        query: inputMessage,
        multiLLM: useMultiLLM,
        activeLLMs: useMultiLLM ? activeLLMs : [],
        expertLLM: useMultiLLM ? expertLLM : "gpt-4",
        fetchChains: false,
        noOfNeighbours: 0  
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if(!response.ok){
        const errorText = await response.text();
        console.error("Error response: ", errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      
      // Process images in the response
      let processedContent = data.response;
      let hasImages = false;

      if (data.images && Object.keys(data.images).length > 0) {
        
        const imageTagRegex = /<image_id>(.*?)<\/image_id>/g;

        processedContent = processedContent.replace(imageTagRegex, (match, imageId) => {
          if (data.images[imageId]){
            hasImages = true;
            return `<img src="data:image/png;base64,${data.images[imageId]}" alt="Retrieved Image" style="max-width: 100%; height: auto; border-radius: 12px; margin: 8px 0; border: 2px solid #00ffff; display: block;" />`;
          } else {
            console.warn(`Image with ID "${imageId}" not found in response`);
            return `<div style="padding: 8px; margin: 8px 0; background-color: #333; border: 1px solid #666; border-radius: 8px; color: #ccc; text-align: center;">Image not found: ${imageId}</div>`;
          }
        });

        // data.images.forEach((imageBase64, index) => {
        //   const imageTag = `<img-${index}>`;
        //   const imgElement = `<img src="data:image/png;base64,${imageBase64}" alt="Generated Image" style="max-width: 100%; height: auto; border-radius: 12px; margin: 8px 0; border: 2px solid #00ffff;" />`;
        //   processedContent = processedContent.replace(imageTag, imgElement);
        // });
      }

      const assistantMessage = { 
        role: 'assistant', 
        content: processedContent,
        reasoning: data.reasoning || '',
        isHtml: hasImages
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: `Sorry, there was an error processing your request: ${error.message}`,
        reasoning: '' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    setUploadedFiles(files);
    setIsUploading(true);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('train_model', trainModel);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const success = await response.json();
      
      if (success) {
        alert('Files uploaded successfully!');
        setUploadedFiles([]);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } else {
        alert('Upload failed. Please try again.');
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const renderSidePanel = () => (
    <div className="w-80 bg-black text-white p-4 flex flex-col h-full border-r border-gray-800 relative overflow-hidden">
      {/* Neon glow background effect */}
      <div className="absolute inset-0 bg-gradient-to-b from-purple-900/20 via-transparent to-cyan-900/20 pointer-events-none"></div>
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500"></div>
      
      <div className="relative z-10">
        <h1 className="text-2xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 animate-pulse">
          ⚡ NEURAL NEXUS
        </h1>
        
        {/* Panel Options */}
        <div className="space-y-3 mb-12">
          <button
            onClick={() => setActivePanel('chat')}
            className={`w-full group relative overflow-hidden p-4 rounded-xl transition-all duration-300 transform hover:scale-105 ${
              activePanel === 'chat' 
                ? 'bg-gradient-to-r from-purple-600 to-pink-600 shadow-lg shadow-purple-500/50' 
                : 'bg-gray-900 hover:bg-gray-800 border border-gray-700 hover:border-purple-500'
            }`}
          >
            <div className="flex items-center space-x-3 relative z-10">
              <MessageCircle size={20} className={activePanel === 'chat' ? 'text-white' : 'text-purple-400'} />
              <span className="font-semibold">UNLEASH AI</span>
            </div>
            {activePanel === 'chat' && (
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 opacity-20 animate-pulse"></div>
            )}
          </button>
          
          <button
            onClick={() => setActivePanel('upload')}
            className={`w-full group relative overflow-hidden p-4 rounded-xl transition-all duration-300 transform hover:scale-105 ${
              activePanel === 'upload' 
                ? 'bg-gradient-to-r from-cyan-600 to-blue-600 shadow-lg shadow-cyan-500/50' 
                : 'bg-gray-900 hover:bg-gray-800 border border-gray-700 hover:border-cyan-500'
            }`}
          >
            <div className="flex items-center space-x-3 relative z-10">
              <Upload size={20} className={activePanel === 'upload' ? 'text-white' : 'text-cyan-400'} />
              <span className="font-semibold">FEED THE BEAST</span>
            </div>
            {activePanel === 'upload' && (
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-blue-600 opacity-20 animate-pulse"></div>
            )}
          </button>
          
          <button
            onClick={() => setActivePanel('analytics')}
            className={`w-full group relative overflow-hidden p-4 rounded-xl transition-all duration-300 transform hover:scale-105 ${
              activePanel === 'analytics' 
                ? 'bg-gradient-to-r from-emerald-600 to-green-600 shadow-lg shadow-emerald-500/50' 
                : 'bg-gray-900 hover:bg-gray-800 border border-gray-700 hover:border-emerald-500'
            }`}
          >
            <div className="flex items-center space-x-3 relative z-10">
              <Database size={20} className={activePanel === 'analytics' ? 'text-white' : 'text-emerald-400'} />
              <span className="font-semibold">NEURAL STATS</span>
            </div>
            {activePanel === 'analytics' && (
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-600 to-green-600 opacity-20 animate-pulse"></div>
            )}
          </button>
        </div>

        {/* LLM Configuration - Psychologically driven */}
        <div className="space-y-4 border-t border-gray-800 pt-6">
          <div className="flex items-center space-x-3 bg-gradient-to-r from-yellow-900/30 to-orange-900/30 p-3 rounded-lg border border-yellow-500/30">
            <div className="relative">
              <input
                type="checkbox"
                id="multiLLM"
                checked={useMultiLLM}
                onChange={(e) => setUseMultiLLM(e.target.checked)}
                className="w-5 h-5 bg-black border-2 border-yellow-500 rounded focus:ring-yellow-500 focus:ring-2 checked:bg-yellow-500"
              />
              {useMultiLLM && <Sparkles size={12} className="absolute -top-1 -right-1 text-yellow-400 animate-spin" />}
            </div>
            <label htmlFor="multiLLM" className="text-yellow-400 font-bold text-sm tracking-wide">
              🔥 ACTIVATE MULTI-BRAIN MODE
            </label>
          </div>

          {useMultiLLM && (
            <div className="space-y-4 animate-fadeIn">
              <div className="bg-gray-900/50 p-4 rounded-lg border border-purple-500/30">
                <label className="block text-purple-300 font-semibold mb-3 text-sm tracking-wide">
                  ⚡ ACTIVE AI WARRIORS
                </label>
                <div className="grid grid-cols-1 gap-2 max-h-40 overflow-y-auto">
                  {availableLLMs.map(llm => (
                    <div key={llm} className="flex items-center space-x-3 p-2 bg-black/50 rounded border border-gray-700 hover:border-purple-500 transition-all">
                      <input
                        type="checkbox"
                        id={llm}
                        checked={activeLLMs.includes(llm)}
                        onChange={() => handleLLMToggle(llm)}
                        className="w-4 h-4 bg-black border-2 border-purple-500 rounded focus:ring-purple-500 focus:ring-2 checked:bg-purple-500"
                      />
                      <label htmlFor={llm} className="text-purple-300 text-sm font-medium flex-1 cursor-pointer">
                        {llm}
                      </label>
                      {activeLLMs.includes(llm) && <Brain size={14} className="text-purple-400 animate-pulse" />}
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-900/50 p-4 rounded-lg border border-cyan-500/30">
                <label className="block text-cyan-300 font-semibold mb-3 text-sm tracking-wide">
                  🧠 SUPREME NEURAL COMMANDER
                </label>
                <select
                  value={expertLLM}
                  onChange={(e) => setExpertLLM(e.target.value)}
                  className="w-full p-3 bg-black border-2 border-cyan-500 rounded-lg text-cyan-300 focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                >
                  <option value="" className="bg-black">Choose Your Digital God</option>
                  {expertLLMOptions.map(option => (
                    <option key={option} value={option} className="bg-black">{option}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderChatPanel = () => (
    <div className="flex-1 flex flex-col h-full bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
      {/* Animated background patterns */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-purple-500/10 via-transparent to-cyan-500/10 animate-pulse"></div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6 space-y-6 relative z-10">
        {messages.length === 0 ? (
          <div className="text-center mt-32 animate-fadeIn">
            <div className="relative mb-8">
              <MessageCircle size={80} className="mx-auto text-purple-500 animate-pulse" />
              <Zap size={24} className="absolute -top-2 -right-2 text-yellow-400 animate-bounce" />
            </div>
            <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400 mb-4">
              READY TO BREAK REALITY?
            </h2>
            <p className="text-gray-400 text-lg">Your AI overlords are waiting for commands...</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-slideIn`}>
              <div className={`max-w-[75%] relative ${
                message.role === 'user' ? 'space-y-0' : 'space-y-3'
              }`}>
                {/* Main message content */}
                <div className={`p-4 rounded-2xl relative overflow-hidden ${
                  message.role === 'user' 
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg shadow-purple-500/30' 
                    : 'bg-gray-800 text-gray-100 border border-gray-700 shadow-lg shadow-cyan-500/20'
                }`}>
                  {message.role === 'assistant' && (
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 to-purple-500"></div>
                  )}
                  {message.isHtml ? (
                    <div 
                      dangerouslySetInnerHTML={{ __html: message.content }}
                      className="prose prose-invert max-w-none"
                    />
                  ) : (
                    <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                  )}
                </div>

                {/* Reasoning section - only for assistant messages */}
                {message.role === 'assistant' && message.reasoning && (
                  <div className="space-y-2">
                    {/* Reasoning toggle button */}
                    <button
                      onClick={() => toggleReasoning(index)}
                      className="flex items-center space-x-2 px-3 py-2 bg-gray-800/50 hover:bg-gray-700/50 border border-gray-600/50 rounded-lg transition-all duration-200 text-gray-300 hover:text-cyan-300 text-sm backdrop-blur-sm"
                    >
                      <Brain size={16} className="text-cyan-400" />
                      <span>AI Reasoning</span>
                      <div className="flex items-center space-x-1 text-xs text-gray-400">
                        {expandedReasoning[index] ? <EyeOff size={14} /> : <Eye size={14} />}
                        {expandedReasoning[index] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      </div>
                    </button>

                    {/* Expandable reasoning content */}
                    <div className={`transition-all duration-300 overflow-hidden ${
                      expandedReasoning[index] ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                    }`}>
                      <div className="bg-gray-900/50 border border-cyan-500/20 rounded-lg p-4 backdrop-blur-sm">
                        <div className="flex items-center space-x-2 mb-3">
                          <Brain size={18} className="text-cyan-400" />
                          <h4 className="text-cyan-300 font-medium">Thought Process</h4>
                        </div>
                        <div className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto custom-scrollbar">
                          {message.reasoning}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex justify-start animate-fadeIn">
            <div className="bg-gray-800 text-gray-100 p-4 rounded-2xl flex items-center space-x-3 border border-cyan-500/30 shadow-lg shadow-cyan-500/20">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
              <span className="text-cyan-300 font-medium">Neural networks thinking...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="p-6 border-t border-gray-800 bg-black/50 backdrop-blur-sm">
        <div className="flex space-x-4">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Command your digital army..."
            className="flex-1 p-4 bg-gray-900 border-2 border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-400 transition-all duration-300"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 shadow-lg shadow-purple-500/50 relative overflow-hidden group"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-pink-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
            <Send size={20} className="relative z-10" />
          </button>
        </div>
      </div>
    </div>
  );

  // const renderUploadPanel = () => (
  //   <div className="flex-1 p-8 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
  //     <div className="absolute inset-0 bg-gradient-to-r from-cyan-900/20 via-transparent to-blue-900/20 pointer-events-none"></div>
      
  //     <div className="relative z-10">
  //       <h2 className="text-3xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">
  //         💾 NEURAL FUEL INJECTION
  //       </h2>
        
  //       <div className="space-y-8">
  //         <div className="bg-gray-900/50 p-6 rounded-xl border border-cyan-500/30 backdrop-blur-sm">
  //           <label className="block text-cyan-300 font-semibold mb-4 text-lg">
  //             ⚡ UPLOAD KNOWLEDGE CORES
  //           </label>
  //           <div className="relative">
  //             <input
  //               type="file"
  //               multiple
  //               ref={fileInputRef}
  //               onChange={handleFileUpload}
  //               className="w-full p-4 bg-black border-2 border-cyan-500 rounded-xl focus:outline-none focus:ring-2 focus:ring-cyan-500 text-cyan-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-600 file:text-white file:font-semibold hover:file:bg-cyan-700 transition-all duration-300"
  //               accept=".pdf,.doc,.docx,.txt,.md"
  //             />
  //             <div className="absolute top-0 right-0 p-2">
  //               <Sparkles size={20} className="text-cyan-400 animate-pulse" />
  //             </div>
  //           </div>
  //         </div>

  //         <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 p-4 rounded-xl border border-orange-500/50">
  //           <div className="flex items-center space-x-3">
  //             <input
  //               type="checkbox"
  //               id="trainModel"
  //               checked={trainModel}
  //               onChange={(e) => setTrainModel(e.target.checked)}
  //               className="w-5 h-5 bg-black border-2 border-orange-500 rounded focus:ring-orange-500 focus:ring-2 checked:bg-orange-500"
  //             />
  //             <label htmlFor="trainModel" className="text-orange-300 font-bold text-sm tracking-wide">
  //               🔥 FORGE NEURAL PATHWAYS (TRAIN BEAST MODE)
  //             </label>
  //           </div>
  //         </div>

  //         {uploadedFiles.length > 0 && (
  //           <div className="bg-gray-900/50 p-6 rounded-xl border border-purple-500/30 backdrop-blur-sm animate-fadeIn">
  //             <h3 className="text-xl font-bold mb-4 text-purple-300">📁 LOADED AMMUNITION:</h3>
  //             <div className="space-y-3">
  //               {uploadedFiles.map((file, index) => (
  //                 <div key={index} className="flex items-center justify-between p-4 bg-black/50 rounded-lg border border-gray-700 hover:border-purple-500 transition-all">
  //                   <div className="flex items-center space-x-3">
  //                     <FileText size={20} className="text-purple-400" />
  //                     <div>
  //                       <span className="text-purple-300 font-medium">{file.name}</span>
  //                       <span className="text-gray-400 text-sm ml-2">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
  //                     </div>
  //                   </div>
  //                   <button
  //                     onClick={() => removeFile(index)}
  //                     className="text-red-400 hover:text-red-300 p-1 hover:bg-red-900/20 rounded transition-all"
  //                   >
  //                     <X size={16} />
  //                   </button>
  //                 </div>
  //               ))}
  //             </div>
  //           </div>
  //         )}

  //         {isUploading && (
  //           <div className="flex items-center justify-center p-12 bg-gray-900/50 rounded-xl border border-cyan-500/30 backdrop-blur-sm animate-pulse">
  //             <div className="text-center">
  //               <div className="flex justify-center mb-4">
  //                 <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
  //               </div>
  //               <p className="text-cyan-300 text-lg font-semibold">INJECTING NEURAL FUEL...</p>
  //               <p className="text-gray-400 text-sm mt-2">Feeding the machine consciousness</p>
  //             </div>
  //           </div>
  //         )}
  //       </div>
  //     </div>
  //   </div>
  // );

  const renderUploadPanel = () => (
    <div className="flex-1 p-8 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-cyan-900/20 via-transparent to-blue-900/20 pointer-events-none"></div>
  
      <div className="relative z-10">
        <h2 className="text-3xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">
          💾 NEURAL FUEL INJECTION
        </h2>
  
        <div className="space-y-8">
          {/* Upload Box */}
          <div className="bg-gray-900/50 p-6 rounded-xl border border-cyan-500/30 backdrop-blur-sm">
            <label className="block text-cyan-300 font-semibold mb-4 text-lg">
              ⚡ UPLOAD KNOWLEDGE CORES
            </label>
            <div className="relative">
              <input
                type="file"
                multiple
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="w-full p-4 bg-black border-2 border-cyan-500 rounded-xl focus:outline-none focus:ring-2 focus:ring-cyan-500 text-cyan-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-600 file:text-white file:font-semibold hover:file:bg-cyan-700 transition-all duration-300"
                accept=".pdf,.doc,.docx,.txt,.md"
              />
              <div className="absolute top-0 right-0 p-2">
                <Sparkles size={20} className="text-cyan-400 animate-pulse" />
              </div>
            </div>
          </div>
  
          {/* Train Checkbox */}
          <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 p-4 rounded-xl border border-orange-500/50">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="trainModel"
                checked={trainModel}
                onChange={(e) => setTrainModel(e.target.checked)}
                className="w-5 h-5 bg-black border-2 border-orange-500 rounded focus:ring-orange-500 focus:ring-2 checked:bg-orange-500"
              />
              <label htmlFor="trainModel" className="text-orange-300 font-bold text-sm tracking-wide">
                🔥 FORGE NEURAL PATHWAYS (TRAIN BEAST MODE)
              </label>
            </div>
          </div>
  
          {/* Uploaded Files List */}
          {uploadedFiles.length > 0 && (
            <div className="bg-gray-900/50 p-6 rounded-xl border border-purple-500/30 backdrop-blur-sm animate-fadeIn">
              <h3 className="text-xl font-bold mb-4 text-purple-300">📁 LOADED AMMUNITION:</h3>
              <div className="space-y-3 max-h-64 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-gray-700">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-black/50 rounded-lg border border-gray-700 hover:border-purple-500 transition-all">
                    <div className="flex items-center space-x-3">
                      <FileText size={20} className="text-purple-400" />
                      <div>
                        <span className="text-purple-300 font-medium">{file.name}</span>
                        <span className="text-gray-400 text-sm ml-2">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="text-red-400 hover:text-red-300 p-1 hover:bg-red-900/20 rounded transition-all"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
  
          {/* Loading Spinner */}
          {isUploading && (
            <div className="flex items-center justify-center p-12 bg-gray-900/50 rounded-xl border border-cyan-500/30 backdrop-blur-sm animate-pulse">
              <div className="text-center">
                <div className="flex justify-center mb-4">
                  <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
                <p className="text-cyan-300 text-lg font-semibold">INJECTING NEURAL FUEL...</p>
                <p className="text-gray-400 text-sm mt-2">Feeding the machine consciousness</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
  

  const renderAnalyticsPanel = () => (
    <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-emerald-900/20 via-transparent to-green-900/20 pointer-events-none"></div>
      
      <div className="text-center relative z-10 animate-fadeIn">
        <div className="relative mb-8">
          <Database size={80} className="mx-auto text-emerald-400 animate-pulse" />
          <div className="absolute -top-2 -right-2 w-6 h-6 bg-gradient-to-r from-emerald-400 to-green-400 rounded-full animate-ping"></div>
        </div>
        <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-green-400 mb-4">
          🧬 NEURAL ANALYTICS LAB
        </h2>
        <p className="text-xl text-gray-400 mb-2">We are cooking something</p>
        <p className="text-emerald-400 font-semibold">ABSOLUTELY MIND-BLOWING</p>
        <div className="mt-8 flex justify-center space-x-2">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-black overflow-hidden">
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
        .animate-slideIn {
          animation: slideIn 0.3s ease-out;
        }
      `}</style>
      
      {renderSidePanel()}
      
      <div className="flex-1 flex flex-col">
        {activePanel === 'chat' && renderChatPanel()}
        {activePanel === 'upload' && renderUploadPanel()}
        {activePanel === 'analytics' && renderAnalyticsPanel()}
      </div>
    </div>
  );
};

export default App;