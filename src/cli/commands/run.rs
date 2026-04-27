//! Agent execution — the main LLM loop.
//!
//! Implements the core agent loop:
//! 1. Assemble context for the turn
//! 2. Call the LLM with tools
//! 3. Parse tool calls from the response
//! 4. Execute tools
//! 5. Apply observation masking (compress old tool results)
//! 6. Feed results back and repeat

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use anyhow::Result;

use crate::cli::commands::agent::display::summarize_args;
use crate::cli::commands::agent::hints::{
    PLAN_CHECKPOINT_AFTER_EDITS, PLAN_CHECKPOINT_BLOCK_MESSAGE, PLAN_CHECKPOINT_WARNING,
    PLAN_HARD_BLOCK_AFTER_EDITS, PLAN_PROGRESS_NUDGE, PREMATURE_EXIT_NUDGE, REPEATED_READ_NUDGE,
    loop_detected_hint, truncated_tool_call_hint,
};
use crate::cli::commands::agent::loop_detector::{is_mutating_call, loop_call_key};
use crate::config::{Config, EditMode, ModelRole};
use crate::context;
use crate::llm::{ChatRequest, Message, ModelRouter, is_truncated_tool_call_error};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::runtime::{
    LlmWorkerEvent, LlmWorkerHandle, ShellControl, ShellWorkerEvent, ToolWorkerPool,
};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui;

/// Run the agent for a single message.
pub async fn run(
    config: Config,
    message: &str,
    plan_only: bool,
    headless: bool,
    continue_session: bool,
) -> Result<()> {
    let log = Arc::new(SessionLog::new(&config));
    log.user_message(message);

    let router = Arc::new(ModelRouter::new(&config));
    let llm_worker = LlmWorkerHandle::new(router.clone(), config.runtime.llm_concurrency);
    let perms = Arc::new(if headless {
        PermissionManager::headless(&config)
    } else {
        PermissionManager::new(&config)
    });
    let tool_pool = ToolWorkerPool::new(config.runtime.tool_worker_pool_size);
    let mut tool_defs = tools::tool_definitions(config.tools.edit_mode);
    // Filter tools based on config
    {
        let mut disabled = Vec::new();
        if !config.tools.web_tools {
            disabled.push("web");
        }
        if !config.tools.plan {
            disabled.push("plan");
        }
        if config.tools.edit_mode == EditMode::Fast {
            // Fast mode replaces edit_file with the primitive surface.
            disabled.push("edit_file");
        }
        tool_defs.retain(|t| !disabled.contains(&t.function.name.as_str()));
        if config.tools.edit_mode == EditMode::Fast {
            tool_defs.extend(tools::fast_mode_tool_definitions());
        }
        tool_defs.push(tools::definitions::spawn_agents_tool_definition());
    }

    // Clear stale scratchpad/plan from previous sessions — unless this
    // is a `--continue` invocation, in which case the model is meant to
    // pick up where the previous session left off.
    if !continue_session {
        let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
        let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
    }

    tui::print_header(if plan_only {
        "Plan Mode (read-only)"
    } else {
        "miniswe"
    });

    // Ask the server what's actually running, so startup reflects reality
    // rather than what config.toml claims. Done once per session.
    for line in router.startup_summary().await {
        tui::print_status(&line);
    }

    // Select model role: plan mode uses the plan model, normal mode uses default
    let model_role = if plan_only {
        ModelRole::Plan
    } else {
        ModelRole::Default
    };

    // Spawn LSP client (non-blocking — initializes in background)
    let lsp_client: Option<Arc<LspClient>> = if config.lsp.enabled {
        match LspClient::spawn(config.project_root.clone()).await {
            Ok(client) => {
                tui::print_status("LSP: starting...");
                Some(Arc::new(client))
            }
            Err(e) => {
                tui::print_status(&format!("LSP: not available ({e})"));
                None
            }
        }
    } else {
        None
    };

    // Initialize MCP servers
    let mcp_config = McpConfig::load(&config.project_root)?;
    let mcp_registry = if mcp_config.has_servers() {
        let cache_dir = config.miniswe_path("mcp");
        match McpRegistry::connect(&mcp_config, &cache_dir) {
            Ok(registry) => {
                if registry.has_servers() {
                    tui::print_status(&format!(
                        "MCP: {} servers, {} tools",
                        registry.servers.len(),
                        registry.tool_count()
                    ));
                    // Add mcp_use tool definition
                    tool_defs.push(tools::definitions::mcp_tool_definition());
                }
                Some(Arc::new(Mutex::new(registry)))
            }
            Err(e) => {
                tui::print_status(&format!("MCP: failed to connect ({e})"));
                None
            }
        }
    } else {
        None
    };

    let mcp_summary = mcp_registry
        .as_ref()
        .and_then(|r| r.lock().context_summary());

    // Estimate tool definition overhead for context budgeting
    let tool_def_tokens =
        context::estimate_tokens(&serde_json::to_string(&tool_defs).unwrap_or_default());

    let max_rounds = config.context.max_rounds;
    let pause_at = config.context.pause_after_rounds;

    let mut conversation_history: Vec<Message> = Vec::new();
    let mut round = 0;
    let mut had_error = false;
    let mut user_continued = false;

    // Track consecutive identical tool calls for loop detection
    let mut last_call_key: Option<String> = None;
    let mut same_call_streak = 0u32;
    // Number of distinct loops the model has been pulled out of in this
    // turn. We give one recovery; a second loop ends the turn for real.
    let mut loop_recoveries = 0u32;
    let mut calls_since_last_edit = 0u32;
    let mut successful_edits_since_plan_update = 0u32;
    let mut plan_checkpoint_pending = false;
    let mut plan_update_requested = false;
    let mut nudged_premature_exit = false;

    // Ctrl+C cancellation flag. The handler fires once and exits — no
    // loop, because `ctrl_c().await` resolves immediately after the
    // first signal and would otherwise busy-spin at 100% CPU.
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_for_handler = cancelled.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        cancelled_for_handler.store(true, Ordering::Relaxed);
        eprintln!("\n\x1b[33m(interrupted — finishing current step)\x1b[0m");
    });

    // Initialize snapshot manager for revert support
    let snapshots = tools::snapshots::SnapshotManager::init(&config.project_root)
        .ok()
        .map(|s| Arc::new(Mutex::new(s)));

    // Fast-mode state: per-file revision store (in-memory, session-scoped)
    // + the project-wide LSP error count captured at session start. The
    // baseline lets each edit's feedback line report `(+N from baseline)`
    // so regressions jump out.
    let fast_revisions: Option<Arc<tools::RevisionStore>> =
        if config.tools.edit_mode == EditMode::Fast {
            let miniswe_dir = config.miniswe_path("revisions");
            match tools::RevisionStore::new(&miniswe_dir) {
                Ok(s) => Some(Arc::new(s)),
                Err(e) => {
                    tui::print_status(&format!("fast mode: revision store init failed ({e})"));
                    None
                }
            }
        } else {
            None
        };
    let fast_baseline_errors: usize = if config.tools.edit_mode == EditMode::Fast {
        tools::fast::project_error_count(lsp_client.as_deref()).await
    } else {
        0
    };

    // Initial context assembly
    let assembled = context::assemble(
        &config,
        message,
        &conversation_history,
        plan_only,
        mcp_summary.as_deref(),
    );
    log.context_assembled(assembled.token_estimate, assembled.messages.len());
    tui::print_status(&format!(
        "Context: ~{} tokens assembled",
        assembled.token_estimate
    ));

    let mut messages = assembled.messages;

    // Nudge the model to plan before editing (if plan tool is available)
    if config.tools.plan {
        messages.push(Message::user(
            "[Before making changes, explore the codebase and use the plan tool to outline your approach. \
             Each step has compile: true (default) — the compiler must pass to check it off. \
             Set compile: false with a reason only if a step intentionally breaks the tree (e.g. renaming a function before updating callers). \
             If a step proves too complex, use action='refine' to split it into substeps. \
             Check off steps as you complete them.]"
        ));
    }

    loop {
        if had_error {
            break;
        }
        round += 1;
        log.round_start(round);

        // Snapshot at start of each round for revert support
        if let Some(ref snap) = snapshots {
            let mut guard = snap.lock();
            let _ = guard.begin_round(round);
        }
        if round > max_rounds {
            tui::print_error("Maximum tool rounds reached. Stopping.");
            break;
        }

        // Ask user if they want to continue after pause_after_rounds
        if round == pause_at && !user_continued {
            tui::print_status(&format!("{pause_at} tool rounds used."));
            let response = tui::read_input("Continue? [y]es / [n]o:");
            match response.as_deref() {
                Some("y") | Some("yes") | Some("") => {
                    user_continued = true;
                }
                _ => {
                    // Tell the LLM to wrap up
                    messages.push(Message::user("[Stop now. Summarize what you've done.]"));
                }
            }
        }

        // Warn the LLM when approaching the hard limit
        if round == max_rounds.saturating_sub(5) {
            messages.push(Message::user(
                "[Approaching tool limit. Wrap up and summarize.]",
            ));
        }

        // Unified context compression — handles both tool results and conversation
        let pre_mask = messages.len();
        context::compressor::maybe_compress(
            &mut messages,
            &config,
            &router,
            &llm_worker,
            tool_def_tokens,
            &mut plan_update_requested,
        )
        .await;
        log.masking_applied(pre_mask.saturating_sub(messages.len()), pre_mask);

        // Sanitize message roles before sending (strict chat template compat)
        context::sanitize_messages(&mut messages);

        // Call LLM with streaming
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.clone()),
            tool_choice: None,
        };
        log.llm_request(&request);

        tui::print_separator();

        // Reset cancel flag for this round
        cancelled.store(false, Ordering::Relaxed);

        eprint!("\x1b[2m⠋ thinking...\x1b[0m");
        std::io::stderr().flush().ok();
        let thinking = Arc::new(AtomicBool::new(true));

        let mut llm_events = llm_worker.submit(model_role, request.clone(), cancelled.clone());
        let response = match loop {
            match llm_events.recv().await {
                Some(LlmWorkerEvent::Token(token)) => {
                    if thinking.load(Ordering::Relaxed) {
                        thinking.store(false, Ordering::Relaxed);
                        eprint!("\r\x1b[2K");
                        std::io::stderr().flush().ok();
                    }
                    tui::print_token(&token);
                }
                Some(LlmWorkerEvent::Completed(Ok(r))) => break Ok(r),
                Some(LlmWorkerEvent::Completed(Err(e))) => break Err(e),
                None => break Err("LLM worker stopped unexpectedly".to_string()),
            }
        } {
            Ok(r) => {
                if thinking.load(Ordering::Relaxed) {
                    eprint!("\r\x1b[2K");
                    std::io::stderr().flush().ok();
                }
                r
            }
            Err(err_str) => {
                eprint!("\r\x1b[2K");
                std::io::stderr().flush().ok();
                if err_str.contains("Interrupted") {
                    tui::print_status("Generation interrupted.");
                    break;
                }
                if is_truncated_tool_call_error(&err_str) {
                    // Model hit max_tokens mid tool-call — the server
                    // dropped the assistant turn, no tool_call_id was
                    // issued. Push a user-role hint and let the agent
                    // retry with a smaller operation.
                    log.llm_error(
                        "tool call JSON truncated (max_tokens) — injecting hint and continuing",
                    );
                    tui::print_status("Previous tool call truncated — retrying with guidance.");
                    let hint = Message::user(truncated_tool_call_hint(config.tools.edit_mode));
                    messages.push(hint.clone());
                    conversation_history.push(hint);
                    continue;
                }
                let clean = if err_str.contains('<') {
                    err_str
                        .split('<')
                        .next()
                        .unwrap_or(&err_str)
                        .trim()
                        .to_string()
                } else {
                    err_str
                };
                log.llm_error(&clean);
                tui::print_error(&format!("LLM error: {clean}"));
                tui::print_status(&format!(
                    "Check that your LLM server is running at {}",
                    config.model.endpoint
                ));
                had_error = true;
                break;
            }
        };

        // Get the assistant's response
        let choice = match response.choices.first() {
            Some(c) => c,
            None => {
                tui::print_error("Empty response from LLM");
                break;
            }
        };

        let assistant_msg = &choice.message;

        // Print newline after streaming content
        if assistant_msg.content.is_some() {
            println!();
        }

        // Log and add assistant message to history
        if let Some(content) = &assistant_msg.content {
            log.llm_response(content);
        }
        if assistant_msg.is_meaningful() {
            conversation_history.push(assistant_msg.clone());
        }

        // Check for tool calls
        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => {
                if !nudged_premature_exit
                    && config.tools.plan
                    && tools::plan::has_unchecked_steps(&config)
                {
                    nudged_premature_exit = true;
                    let nudge = Message::user(PREMATURE_EXIT_NUDGE);
                    messages.push(nudge.clone());
                    conversation_history.push(nudge);
                    continue;
                }
                break;
            }
        };

        // Add assistant's tool call message to messages
        messages.push(assistant_msg.clone());

        for tc in &tool_calls {
            let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    let result_msg = Message::tool_result(
                        &tc.id,
                        &format!(
                            "Invalid JSON in tool arguments: {e}\nRaw: {}",
                            tc.function.arguments
                        ),
                    );
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    tui::print_tool_result(&tc.function.name, false, "invalid JSON args");
                    continue;
                }
            };

            let args_summary = summarize_args(&tc.function.name, &args);

            // Detect tool call loops: only identical calls repeated consecutively.
            let call_key = loop_call_key(&tc.function.name, &args);
            if last_call_key.as_ref() == Some(&call_key) {
                same_call_streak += 1;
            } else {
                last_call_key = Some(call_key.clone());
                same_call_streak = 1;
            }
            if same_call_streak >= 3 {
                let mutating = is_mutating_call(&tc.function.name, &args);
                log.loop_detected(&tc.function.name, &args_summary, same_call_streak as usize);

                // Read-only repetition: harmless, just wasted tokens.
                // Surface a polite nudge inline and let the for-loop
                // continue so the model can react in subsequent rounds.
                if !mutating {
                    let result_msg = Message::tool_result(&tc.id, REPEATED_READ_NUDGE);
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    tui::print_status(&format!(
                        "Repeated read: {}({}) — nudge sent, continuing",
                        tc.function.name, args_summary
                    ));
                    last_call_key = None;
                    same_call_streak = 0;
                    continue;
                }

                let result_msg =
                    Message::tool_result(&tc.id, loop_detected_hint(config.tools.edit_mode));
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);

                // First mutating loop in this turn: surface the hint, reset
                // the streak, and let the model try a different approach.
                // Subsequent loops mean the recovery itself spiraled —
                // abort for real.
                if loop_recoveries == 0 {
                    loop_recoveries += 1;
                    last_call_key = None;
                    same_call_streak = 0;
                    tui::print_error(&format!(
                        "Loop detected: {}({}) repeated 3 times — surfacing a hint, giving the model one more round",
                        tc.function.name, args_summary
                    ));
                    break;
                }
                tui::print_error(&format!(
                    "Loop detected again ({}({})) after the recovery hint — stopping this turn",
                    tc.function.name, args_summary
                ));
                had_error = true;
                break;
            }

            log.tool_call_detail(&tc.function.name, &args);
            tui::print_tool_call(&tc.function.name, &args_summary);

            // Block write tools in plan-only mode
            let file_action = args["action"].as_str().unwrap_or("");
            if plan_only
                && ((tc.function.name == "file" && file_action == "shell")
                    || matches!(tc.function.name.as_str(), "edit_file" | "write_file"))
            {
                let result_msg = Message::tool_result(
                    &tc.id,
                    "Blocked: plan mode is read-only. No edits or shell commands allowed.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                tui::print_tool_result(&tc.function.name, false, "blocked in plan mode");
                continue;
            }

            // Write gating: require plan before write tools
            let is_write_action = matches!(tc.function.name.as_str(), "edit_file" | "write_file");
            if config.tools.plan && !tools::plan::plan_exists(&config) && is_write_action {
                let result_msg = Message::tool_result(
                    &tc.id,
                    "Create a plan first: use plan(action='set') with your step-by-step approach before making changes.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                tui::print_tool_result(&tc.function.name, false, "blocked: no plan");
                continue;
            }
            if config.tools.plan
                && plan_checkpoint_pending
                && successful_edits_since_plan_update >= PLAN_HARD_BLOCK_AFTER_EDITS
                && is_write_action
            {
                let result_msg = Message::tool_result(&tc.id, PLAN_CHECKPOINT_BLOCK_MESSAGE);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                tui::print_tool_result(&tc.function.name, false, "blocked: plan checkpoint");
                continue;
            }

            // Handle tool dispatch
            let mut result = if tc.function.name == "file" && file_action == "revert" {
                let snapshots = snapshots.clone();
                let args = args.clone();
                match tool_pool
                    .submit(move || {
                        let to_round = args["to_round"].as_u64().unwrap_or(0) as usize;
                        let path = args["path"].as_str().unwrap_or("").to_string();
                        match snapshots {
                            Some(snap) => {
                                let guard = snap.lock();
                                let res = if !path.is_empty() {
                                    guard.revert_file(&path, to_round)
                                } else {
                                    guard.revert_to_round(to_round)
                                };
                                res.map(crate::tools::ToolResult::ok)
                                    .map_err(|e| format!("Revert failed: {e}"))
                            }
                            None => Ok(crate::tools::ToolResult::err(
                                "Snapshot system not available (git not found?)".into(),
                            )),
                        }
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => {
                        crate::tools::ToolResult::err("Tool worker dropped revert job".into())
                    }
                }
            } else if tc.function.name == "plan" {
                let args = args.clone();
                let config = config.clone();
                match tool_pool
                    .submit(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .map_err(|e| e.to_string())?;
                        runtime
                            .block_on(
                                async move { tools::plan::execute(&args, &config, round).await },
                            )
                            .map_err(|e| format!("plan error: {e}"))
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => crate::tools::ToolResult::err("Tool worker dropped plan job".into()),
                }
            } else if tc.function.name == "edit_file" {
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let router = router.clone();
                let lsp = lsp_client.clone();
                let cancelled = cancelled.clone();
                let log = log.clone();
                match tool_pool
                    .submit(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .map_err(|e| e.to_string())?;
                        runtime
                            .block_on(async move {
                                tools::execute_edit_file_tool(
                                    &args,
                                    &config,
                                    perms.as_ref(),
                                    router.as_ref(),
                                    lsp.as_deref(),
                                    Some(cancelled.as_ref()),
                                    Some(log.as_ref()),
                                )
                                .await
                            })
                            .map_err(|e| format!("edit_file error: {e}"))
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => {
                        crate::tools::ToolResult::err("Tool worker dropped edit_file job".into())
                    }
                }
            } else if tc.function.name == "file" && file_action == "shell" {
                await_shell_job_run(
                    tool_pool.submit_shell(args.clone(), config.clone(), cancelled.clone()),
                    cancelled.as_ref(),
                )
                .await
            } else if matches!(
                tc.function.name.as_str(),
                "replace_range" | "insert_at" | "revert" | "show_rev" | "check"
            ) && config.tools.edit_mode == EditMode::Fast
            {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp_client.clone();
                let revisions = fast_revisions.clone();
                let baseline = fast_baseline_errors;
                match tool_pool
                    .submit(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .map_err(|e| e.to_string())?;
                        let Some(revisions) = revisions else {
                            return Ok(crate::tools::ToolResult::err(
                                "fast mode: revision store unavailable".into(),
                            ));
                        };
                        runtime
                            .block_on(async move {
                                tools::execute_fast_tool(
                                    &tool_name,
                                    &args,
                                    &config,
                                    perms.as_ref(),
                                    lsp.as_deref(),
                                    revisions.as_ref(),
                                    baseline,
                                )
                                .await
                            })
                            .map_err(|e| format!("fast tool error: {e}"))
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => {
                        crate::tools::ToolResult::err("Tool worker dropped fast tool job".into())
                    }
                }
            } else if tc.function.name == "mcp_use" {
                let server = args["server"].as_str().unwrap_or("").to_string();
                let tool = args["tool"].as_str().unwrap_or("").to_string();
                let tool_args = args.get("arguments").cloned().unwrap_or_default();
                if server.is_empty() || tool.is_empty() {
                    crate::tools::ToolResult::err(
                        "mcp_use requires top-level 'server' and 'tool' string fields. \
                         Example: {\"server\": \"my-server\", \"tool\": \"my-tool\", \"arguments\": {}}".into(),
                    )
                } else {
                    match perms.check(&Action::McpUse(server.clone(), tool.clone())) {
                        Err(e) => crate::tools::ToolResult::err(e),
                        Ok(()) => {
                            let registry = mcp_registry.clone();
                            match tool_pool
                                .submit(move || match registry {
                                    Some(registry) => {
                                        let mut guard = registry.lock();
                                        guard
                                            .call_tool(&server, &tool, tool_args)
                                            .map(crate::tools::ToolResult::ok)
                                            .map_err(|e| format!("MCP error: {e}"))
                                    }
                                    None => Ok(crate::tools::ToolResult::err(
                                        "No MCP servers connected".into(),
                                    )),
                                })
                                .await
                            {
                                Ok(Ok(r)) => r,
                                Ok(Err(e)) => crate::tools::ToolResult::err(e),
                                Err(_) => crate::tools::ToolResult::err(
                                    "Tool worker dropped mcp job".into(),
                                ),
                            }
                        }
                    }
                }
            } else if tc.function.name == "spawn_agents" {
                let tasks = crate::cli::commands::agent::subagent::parse_tasks(&args);
                if tasks.is_empty() {
                    crate::tools::ToolResult::err(
                        "spawn_agents: 'agents' must be a non-empty array of {label, prompt}"
                            .into(),
                    )
                } else {
                    tui::print_status(&format!("spawning {} subagents...", tasks.len()));
                    let outputs = crate::cli::commands::agent::subagent::run_subagents(
                        tasks,
                        &config,
                        &llm_worker,
                        &tool_pool,
                        &tool_defs,
                        &perms,
                        &mcp_registry,
                        &lsp_client,
                        &fast_revisions,
                        fast_baseline_errors,
                        &cancelled,
                        None,
                    )
                    .await;
                    let combined = crate::cli::commands::agent::subagent::format_outputs(outputs);
                    crate::tools::ToolResult::ok(combined)
                }
            } else {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp_client.clone();
                match tool_pool
                    .submit(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .map_err(|e| e.to_string())?;
                        runtime
                            .block_on(async move {
                                tools::execute_tool(
                                    &tool_name,
                                    &args,
                                    &config,
                                    perms.as_ref(),
                                    lsp.as_deref(),
                                )
                                .await
                            })
                            .map_err(|e| format!("Tool error: {e}"))
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => crate::tools::ToolResult::err("Tool worker dropped job".into()),
                }
            };

            if !result.success
                && let Some(hint) = tools::plan::failure_hint(&config)
            {
                result.content.push('\n');
                result.content.push_str(&hint);
            }

            // Append round number to every tool result
            result
                .content
                .push_str(&format!("\n[round {round}/{max_rounds}]"));

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            tui::print_tool_result(&tc.function.name, result.success, first_line);

            if result.success && tc.function.name == "plan" {
                plan_checkpoint_pending = false;
                successful_edits_since_plan_update = 0;
            }

            // A successful file write means code changed — reset trackers.
            let is_file_write = matches!(tc.function.name.as_str(), "edit_file" | "write_file");
            if result.success && is_file_write {
                last_call_key = None;
                same_call_streak = 0;
                calls_since_last_edit = 0;
                if config.tools.plan {
                    if tools::plan::plan_exists(&config) {
                        result.content.push('\n');
                        result.content.push_str(PLAN_PROGRESS_NUDGE);
                    }
                    successful_edits_since_plan_update += 1;
                    if successful_edits_since_plan_update >= PLAN_CHECKPOINT_AFTER_EDITS {
                        plan_checkpoint_pending = true;
                    }
                    if successful_edits_since_plan_update == PLAN_CHECKPOINT_AFTER_EDITS {
                        result.content.push('\n');
                        result.content.push_str(PLAN_CHECKPOINT_WARNING);
                    }
                }
            } else {
                calls_since_last_edit += 1;
            }

            let result_msg = Message::tool_result(&tc.id, &result.content);
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);
        }

        // Stall detection: too many tool calls without any edits
        if calls_since_last_edit >= 20 && calls_since_last_edit.is_multiple_of(20) {
            let edit_hint = match config.tools.edit_mode {
                EditMode::Smart => "Use edit_file for semantic file edits.",
                EditMode::Fast => "Use replace_range or insert_at to land targeted edits.",
            };
            messages.push(Message::user(&format!(
                "[WARNING: You have used 20+ tool calls without making any edits. \
                 You likely have enough information. Start making changes now. \
                 {edit_hint} \
                 If you're stuck, explain what's blocking you.]"
            )));
        }
    }

    log.session_end(round, had_error);

    // Shut down LSP
    if let Some(lsp) = lsp_client
        && let Ok(lsp) = Arc::try_unwrap(lsp)
    {
        lsp.shutdown().await;
    }

    tui::print_separator();
    if !had_error {
        tui::print_complete("Done");
    }

    Ok(())
}

async fn await_shell_job_run(
    mut shell_job: crate::runtime::ShellJobHandle,
    cancelled: &AtomicBool,
) -> crate::tools::ToolResult {
    while let Some(event) = shell_job.events_rx.recv().await {
        match event {
            ShellWorkerEvent::TimedOut {
                command,
                timeout_secs,
            } => {
                let prompt = format!(
                    "Shell command still running after {timeout_secs}s.\n  $ {command}\n[c]ontinue waiting / [k]ill: "
                );
                let response = crate::tui::read_input(&prompt)
                    .unwrap_or_else(|| "k".into())
                    .trim()
                    .to_lowercase();
                let control = if response == "c" || response == "continue" {
                    crate::tui::print_status("continuing to wait for shell command...");
                    ShellControl::Continue
                } else {
                    ShellControl::Kill
                };
                if shell_job.send_control(control).is_err() {
                    return crate::tools::ToolResult::err(
                        "Shell worker dropped control channel".into(),
                    );
                }
            }
            ShellWorkerEvent::Completed(result) => {
                if cancelled.load(Ordering::Relaxed) {
                    cancelled.store(false, Ordering::Relaxed);
                }
                return match result {
                    Ok(tool_result) => tool_result,
                    Err(err) => crate::tools::ToolResult::err(err),
                };
            }
        }
    }
    if cancelled.load(Ordering::Relaxed) {
        cancelled.store(false, Ordering::Relaxed);
    }
    crate::tools::ToolResult::err("Shell worker dropped before reporting a result".into())
}
