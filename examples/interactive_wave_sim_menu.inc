// =============================================================================
// Menu System - Callbacks, Initialization, and Management
// This file is designed to be #include'd into interactive_wave_sim.c
// All necessary types and functions should already be available from the parent file
// =============================================================================

// Global dummy variables for button callbacks (must persist beyond initialization)
static int g_dummy_reset = 0;
static int g_dummy_clear_barriers = 0;
static int g_dummy_clear_sources = 0;
static int g_dummy_delete_source = 0;
static int g_dummy_deselect_source = 0;

// =============================================================================
// Menu Helper Functions for Centralized Drag Handling
// =============================================================================

// Check if interaction is a slider
int is_slider_interaction(const VariableInteraction *vi) {
    return vi && vi->type == VAR_SLIDER;
}

// Find which interaction was clicked in a menu
VariableInteraction* menu_get_interaction_at(Menu *menu, int mx, int my) {
    if (!menu) return NULL;
    
    // Check if click is within menu bounds
    if (mx < menu->x || mx > menu->x + menu->width ||
        my < menu->y || my > menu->y + menu->height) {
        return NULL;
    }
    
    int num_rows = menu->rows ? menu->rows->size : 0;
    if (num_rows == 0) return NULL;
    
    int title_h = 24;
    int pad = 8;
    int gap = 8;
    int row_h = (menu->height - title_h) / num_rows;
    
    for (int row_idx = 0; row_idx < num_rows; row_idx++) {
        MenuRow *row = (MenuRow*)dynarray_get(menu->rows, row_idx);
        if (!row) continue;
        
        int ry = menu->y + title_h + row_idx * row_h;
        
        // Check if click is in this row's vertical bounds
        if (my < ry || my >= ry + row_h) continue;
        
        int num_cols = row->interactions ? row->interactions->size : 0;
        if (num_cols == 0) continue;
        
        int slot_w = menu->width / num_cols;
        
        for (int col_idx = 0; col_idx < num_cols; col_idx++) {
            VariableInteraction *vi = (VariableInteraction*)dynarray_get(row->interactions, col_idx);
            if (!vi || !is_slider_interaction(vi)) continue;
            
            int rx = menu->x + col_idx * slot_w;
            
            // Calculate slider bar position (matching render logic)
            int label_w = strlen(vi->name) * 7;  // Approximate label width
            int bar_x = rx + pad + label_w + gap;
            int bar_w = slot_w - (pad + label_w + gap + pad);
            if (bar_w < 32) bar_w = 32;
            
            int bar_h = 12;
            int bar_y = ry + row_h/2 - 6;  // Center bar in row
            
            // Check if click is within slider bar bounds
            if (mx >= bar_x && mx <= bar_x + bar_w &&
                my >= bar_y && my <= bar_y + bar_h) {
                return vi;
            }
        }
    }
    return NULL;
}

// Start dragging a slider
void menu_begin_slider_drag(Menu *menu, VariableInteraction *vi, int mx) {
    if (!vi || !is_slider_interaction(vi)) return;
    
    g_menu_drag_state.dragging_interaction = vi;
    g_menu_drag_state.drag_start_x = mx;
    g_menu_drag_state.drag_start_value = *(double*)vi->variable;
    g_menu_drag_state.drag_source_menu = menu;
}

// Update dragging slider
void menu_update_slider_drag(int mx, int window_width) {
    (void)window_width;  // May be used for clamping in future
    
    if (!g_menu_drag_state.dragging_interaction) return;
    
    VariableInteraction *vi = g_menu_drag_state.dragging_interaction;
    Menu *menu = g_menu_drag_state.drag_source_menu;
    if (!menu) return;
    
    // Find which row/col contains this interaction
    int num_rows = menu->rows ? menu->rows->size : 0;
    int pad = 8;
    int gap = 8;
    
    for (int row_idx = 0; row_idx < num_rows; row_idx++) {
        MenuRow *row = (MenuRow*)dynarray_get(menu->rows, row_idx);
        if (!row) continue;
        
        int num_cols = row->interactions ? row->interactions->size : 0;
        int slot_w = menu->width / num_cols;
        
        for (int col_idx = 0; col_idx < num_cols; col_idx++) {
            VariableInteraction *check_vi = (VariableInteraction*)dynarray_get(row->interactions, col_idx);
            if (check_vi != vi) continue;
            
            // Found the interaction - calculate slider bar width
            int label_w = strlen(vi->name) * 7;
            int bar_w = slot_w - (pad + label_w + gap + pad);
            if (bar_w < 32) bar_w = 32;
            
            // Convert mouse delta to value delta
            int delta_x = mx - g_menu_drag_state.drag_start_x;
            double value_range = vi->max - vi->min;
            double delta_value = ((double)delta_x / (double)bar_w) * value_range;
            
            // Calculate new value
            double new_value = g_menu_drag_state.drag_start_value + delta_value;
            
            // Clamp to bounds
            if (new_value < vi->min) new_value = vi->min;
            if (new_value > vi->max) new_value = vi->max;
            
            // Update the variable
            *(double*)vi->variable = new_value;
            
            // Call callback if present
            if (vi->on_change) {
                vi->on_change(vi, vi->callback_data);
            }
            
            return;
        }
    }
}

// End dragging
void menu_end_slider_drag(void) {
    g_menu_drag_state.dragging_interaction = NULL;
    g_menu_drag_state.drag_source_menu = NULL;
}

// =============================================================================
// Menu Callbacks
// =============================================================================

void on_pause_toggle(VariableInteraction *vi, void *user_data) {
    (void)vi;  // Unused
    SimulationState *sim = (SimulationState*)user_data;
    printf("Simulation %s\n", sim->paused ? "paused" : "resumed");
}

void on_reset_clicked(VariableInteraction *vi, void *user_data) {
    (void)vi;  // Unused
    SimulationState *sim = (SimulationState*)user_data;
    reset_simulation(sim);
    printf("Simulation reset\n");
}

void on_timestep_changed(VariableInteraction *vi, void *user_data) {
    (void)vi;  // Unused
    SimulationState *sim = (SimulationState*)user_data;
    /* Estimate a CFL using the configured impulse strength as a proxy for max velocity */
    double cfl = sim->velocity_impulse_strength * sim->dt / fmin(sim->dx, sim->dy);
    if (cfl > 0.7) {
        printf("WARNING: CFL = %.4f may be unstable!\n", cfl);
    }
}

// Radio button callbacks for render mode
void on_render_pressure_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    RenderState *render = (RenderState*)user_data;
    if (render->mode_pressure) {
        render->mode = RENDER_PRESSURE;
        render->mode_pressure = 1;
        render->mode_velocity = 0;
        render->mode_vorticity = 0;
        render->mode_rgb = 0;
        printf("Render mode: PRESSURE\n");
    }
}

void on_render_vorticity_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    RenderState *render = (RenderState*)user_data;
    if (render->mode_vorticity) {
        render->mode = RENDER_VORTICITY;
        render->mode_pressure = 0;
        render->mode_velocity = 0;
        render->mode_vorticity = 1;
        render->mode_rgb = 0;
        printf("Render mode: VORTICITY\n");
    }
}

void on_render_velocity_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    RenderState *render = (RenderState*)user_data;
    if (render->mode_velocity) {
        render->mode = RENDER_VELOCITY;
    render->mode_pressure = 0;
        render->mode_velocity = 1;
    render->mode_vorticity = 0;
    render->mode_rgb = 0;
        printf("Render mode: VELOCITY\n");
    }
}

void on_render_rgb_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    RenderState *render = (RenderState*)user_data;
    if (render->mode_rgb) {
        render->mode = RENDER_RGB;
    render->mode_pressure = 0;
        render->mode_velocity = 0;
        render->mode_rgb = 1;
        printf("Render mode: RGB\n");
    }
}

// Radio button callbacks for mouse mode
void on_mouse_none_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    if (g_mouse_none) {
        g_mouse_mode = MOUSE_NONE;
        g_mouse_none = 1;
    g_mouse_add_velocity = 0;
        g_mouse_add_barrier = 0;
        g_mouse_source = 0;
        
        // Clear unpaired points only
        if (n_barrier_points % 2 == 1) {
            n_barrier_points--;
        }
        
        // Reset dragging state
        dragging_barrier_point = -1;
        dragging_barrier_idx = -1;
        barrier_click_candidate = -1;
        
        printf("Mode: NONE\n");
    }
}

void on_mouse_add_wave_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    if (g_mouse_add_velocity) {
        g_mouse_mode = MOUSE_ADD_VELOCITY;
        g_mouse_none = 0;
        g_mouse_add_velocity = 1;
        g_mouse_add_barrier = 0;
        g_mouse_source = 0;
        
        // Clear unpaired points only
        if (n_barrier_points % 2 == 1) {
            n_barrier_points--;
        }
        
        // Reset dragging state
        dragging_barrier_point = -1;
        dragging_barrier_idx = -1;
        barrier_click_candidate = -1;
        
    printf("Mode: ADD_VELOCITY\n");
    }
}

void on_mouse_add_barrier_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    if (g_mouse_add_barrier) {
        g_mouse_mode = MOUSE_ADD_BARRIER;
        g_mouse_none = 0;
    g_mouse_add_velocity = 0;
        g_mouse_add_barrier = 1;
        g_mouse_source = 0;
        
        // Only clear unpaired points (odd count), keep complete barriers
        if (n_barrier_points % 2 == 1) {
            n_barrier_points--;
            printf("Cleared unpaired barrier point\n");
        }
        
        // Reset dragging state
        dragging_barrier_point = -1;
        dragging_barrier_idx = -1;
        barrier_click_candidate = -1;
        
        printf("Mode: BARRIER\n");
    }
}

void on_mouse_source_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    if (g_mouse_source) {
        g_mouse_mode = MOUSE_SOURCE;
        g_mouse_none = 0;
    g_mouse_add_velocity = 0;
        g_mouse_add_barrier = 0;
        g_mouse_source = 1;
        
        // Clear unpaired points only
        if (n_barrier_points % 2 == 1) {
            n_barrier_points--;
        }
        
        // Reset dragging state
        dragging_barrier_point = -1;
        dragging_barrier_idx = -1;
        barrier_click_candidate = -1;
        
        printf("Mode: SOURCE\n");
    }
}

void on_clear_barriers_clicked(VariableInteraction *vi, void *user_data) {
    (void)vi;
    SimulationState *sim = (SimulationState *)user_data;
    
    // Clear all interior boundaries from the grid
    if (sim && sim->grid) {
        sim->grid->n_interior_boundaries = 0;
        printf("Cleared all barriers\n");
    }
    
    // Clear temporary barrier points
    n_barrier_points = 0;
}

void on_toggle_mouse_controls(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    // Menu system already toggled g_show_mouse_controls
    // Just handle mutual exclusion
    if (g_show_mouse_controls) {
        g_show_sim_controls = 0;  // Close sim controls when opening mouse controls
        printf("Mouse controls menu opened\n");
    } else {
        printf("Mouse controls menu closed\n");
    }
}

void on_delete_source_clicked(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    
    if (selected_source && selected_source->active) {
        int idx = selected_source - wave_sources;
        selected_source->active = 0;  // Mark as inactive
        printf("Deleted source #%d\n", idx);
        selected_source = NULL;  // Deselect
        last_menu_source = NULL;  // Clear menu tracking
    }
}

void on_deselect_source_clicked(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    
    if (selected_source) {
        int idx = selected_source - wave_sources;
        printf("Deselected source #%d\n", idx);
        selected_source = NULL;  // Deselect
        last_menu_source = NULL;  // Clear menu tracking
    }
}

void on_clear_sources_clicked(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    
    // Deactivate all sources
    for (int i = 0; i < n_wave_sources; i++) {
        wave_sources[i].active = 0;
    }
    n_wave_sources = 0;
    selected_source = NULL;
    last_menu_source = NULL;  // Clear menu tracking
    printf("Cleared all sources\n");
}

void on_toggle_sim_controls(VariableInteraction *vi, void *user_data) {
    (void)vi;
    (void)user_data;
    // Menu system already toggled g_show_sim_controls
    // Just handle mutual exclusion
    if (g_show_sim_controls) {
        g_show_mouse_controls = 0;  // Close mouse controls when opening sim controls
        printf("Simulation controls menu opened\n");
    } else {
        printf("Simulation controls menu closed\n");
    }
}

// =============================================================================
// Menu Structure and Initialization
// =============================================================================

typedef struct {
    Menu *base_menu;
    Menu *mouse_menu;
    Menu *sim_menu;
    Menu *source_menu;
} Menus;

Menus* initialize_menus(SimulationState *sim, RenderState *render) {
    Menus *menus = (Menus*)malloc(sizeof(Menus));
    
    Color textColor = {255, 255, 255, 255};
    Color bgColor = {30, 30, 40, 220};
    
    // Base menu - just toggle buttons for submenus
    menus->base_menu = menu_create(10, 10, 250, 150, 1, "Controls", textColor, bgColor);
    MenuRow *base_row1 = menurow_create();
    menurow_add_interaction(base_row1,
        variableinteraction_create(&g_show_mouse_controls, "Mouse Controls", 0, 1, VAR_BOOL,
                                  on_toggle_mouse_controls, NULL));
    menu_add_row(menus->base_menu, base_row1);
    
    MenuRow *base_row2 = menurow_create();
    menurow_add_interaction(base_row2,
        variableinteraction_create(&g_show_sim_controls, "Sim Controls", 0, 1, VAR_BOOL,
                                  on_toggle_sim_controls, NULL));
    menu_add_row(menus->base_menu, base_row2);
    
    // Mouse controls submenu - 2x2 grid for mode toggles
    menus->mouse_menu = menu_create(270, 10, 250, 230, 1, "Mouse Controls", textColor, bgColor);
    
    // Row 1: None and Add Velocity
    MenuRow *m_row1 = menurow_create();
    menurow_add_interaction(m_row1,
        variableinteraction_create(&g_mouse_none, "None", 0, 1, VAR_BOOL,
                                  on_mouse_none_selected, NULL));
    menurow_add_interaction(m_row1,
        variableinteraction_create(&g_mouse_add_velocity, "Add Velocity", 0, 1, VAR_BOOL,
                                  on_mouse_add_wave_selected, NULL));
    menu_add_row(menus->mouse_menu, m_row1);
    
    // Row 2: Barrier and Source
    MenuRow *m_row2 = menurow_create();
    menurow_add_interaction(m_row2,
        variableinteraction_create(&g_mouse_add_barrier, "Barrier", 0, 1, VAR_BOOL,
                                  on_mouse_add_barrier_selected, NULL));
    menurow_add_interaction(m_row2,
        variableinteraction_create(&g_mouse_source, "Source", 0, 1, VAR_BOOL,
                                  on_mouse_source_selected, NULL));
    menu_add_row(menus->mouse_menu, m_row2);
    
    // Row 3: Impulse Strength slider
    MenuRow *m_row3 = menurow_create();
    menurow_add_interaction(m_row3,
        variableinteraction_create(&sim->velocity_impulse_strength, "Impulse Strength", 0.0, 50.0, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(menus->mouse_menu, m_row3);
    
    // Row 4: Spread slider
    MenuRow *m_row4 = menurow_create();
    menurow_add_interaction(m_row4,
        variableinteraction_create(&sim->velocity_impulse_radius, "Impulse Radius", 0.001, 1.0, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(menus->mouse_menu, m_row4);
    
    // Row 5: Source Frequency slider
    MenuRow *m_row5 = menurow_create();
    menurow_add_interaction(m_row5,
        variableinteraction_create(&g_default_source_frequency, "Source Freq (Hz)", 0.5, 50.0, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(menus->mouse_menu, m_row5);
    
    // Row 6: Source Phase slider
    MenuRow *m_row6 = menurow_create();
    menurow_add_interaction(m_row6,
        variableinteraction_create(&g_default_source_phase, "Source Phase (rad)", 0.0, 6.28, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(menus->mouse_menu, m_row6);
    
    // Source control submenu (shown when source is selected)
    menus->source_menu = menu_create(530, 10, 250, 220, 1, "Source Controls", textColor, bgColor);
    
    // We'll populate this dynamically based on selected source
    // For now, create placeholders - will be updated when source is selected
    
    // Sim controls submenu
    menus->sim_menu = menu_create(270, 10, 250, 340, 1, "Simulation Controls", textColor, bgColor);
    
    MenuRow *s_row1 = menurow_create();
    menurow_add_interaction(s_row1,
        variableinteraction_create(&sim->paused, "Paused", 0, 1, VAR_BOOL,
                                  on_pause_toggle, sim));
    menu_add_row(menus->sim_menu, s_row1);
    
    MenuRow *s_row2 = menurow_create();
    menurow_add_interaction(s_row2,
        variableinteraction_create(&g_dummy_reset, "Reset", 0, 1, VAR_BOOL,
                                  on_reset_clicked, sim));
    menu_add_row(menus->sim_menu, s_row2);
    
    MenuRow *s_row3 = menurow_create();
    menurow_add_interaction(s_row3,
    variableinteraction_create(&sim->viscosity, "Viscosity", 0.0, 1.0, VAR_SLIDER,
                  NULL, NULL));
    menu_add_row(menus->sim_menu, s_row3);
    
    MenuRow *s_row4 = menurow_create();
    menurow_add_interaction(s_row4,
        variableinteraction_create(&sim->max_sim_speed, "Max Steps/s", 100.0, 5000.0, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(menus->sim_menu, s_row4);
    
    MenuRow *s_row5 = menurow_create();
    menurow_add_interaction(s_row5,
        variableinteraction_create(&render->value_scale, "Scale", 0.01, 20.0, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(menus->sim_menu, s_row5);
    
    MenuRow *s_row6 = menurow_create();
    menurow_add_interaction(s_row6,
        variableinteraction_create(&g_dummy_clear_barriers, "Clear Barriers", 0, 1, VAR_BOOL,
                                  on_clear_barriers_clicked, sim));
    menu_add_row(menus->sim_menu, s_row6);
    
    MenuRow *s_row7 = menurow_create();
    menurow_add_interaction(s_row7,
        variableinteraction_create(&g_dummy_clear_sources, "Clear Sources", 0, 1, VAR_BOOL,
                                  on_clear_sources_clicked, NULL));
    menu_add_row(menus->sim_menu, s_row7);
    
    MenuRow *s_row8 = menurow_create();
    menurow_add_interaction(s_row8,
        variableinteraction_create(&render->show_boundaries, "Show Boundaries", 0, 1, VAR_BOOL,
                                  NULL, NULL));
    menu_add_row(menus->sim_menu, s_row8);
    
    MenuRow *s_row9 = menurow_create();
    menurow_add_interaction(s_row9,
        variableinteraction_create(&render->show_stats, "Show Stats", 0, 1, VAR_BOOL,
                                  NULL, NULL));
    menu_add_row(menus->sim_menu, s_row9);
    
    MenuRow *s_row10 = menurow_create();
    menurow_add_interaction(s_row10,
        variableinteraction_create(&render->mode_pressure, "Mode: Pressure", 0, 1, VAR_BOOL,
                                  on_render_pressure_selected, render));
    menu_add_row(menus->sim_menu, s_row10);

    MenuRow *s_row11 = menurow_create();
    menurow_add_interaction(s_row11,
        variableinteraction_create(&render->mode_velocity, "Mode: Velocity", 0, 1, VAR_BOOL,
                                  on_render_velocity_selected, render));
    menu_add_row(menus->sim_menu, s_row11);

    MenuRow *s_row12 = menurow_create();
    menurow_add_interaction(s_row12,
        variableinteraction_create(&render->mode_vorticity, "Mode: Vorticity", 0, 1, VAR_BOOL,
                                  on_render_vorticity_selected, render));
    menu_add_row(menus->sim_menu, s_row12);

    MenuRow *s_row13 = menurow_create();
    menurow_add_interaction(s_row13,
        variableinteraction_create(&render->mode_rgb, "Mode: RGB", 0, 1, VAR_BOOL,
                                  on_render_rgb_selected, render));
    menu_add_row(menus->sim_menu, s_row13);
    
    return menus;
}

void update_source_menu(Menu **source_menu_ptr, WaveSource *source) {
    // Free existing menu
    if (*source_menu_ptr) {
        // Clear drag state if it's pointing to this menu's interactions
        if (g_menu_drag_state.drag_source_menu == *source_menu_ptr) {
            menu_end_slider_drag();
        }
        menu_free(*source_menu_ptr);
        *source_menu_ptr = NULL;
    }
    
    if (!source || !source->active) {
        return;  // Menu will be hidden when no source selected
    }
    
    // Create new menu with current source parameters
    Color textColor = {255, 255, 255, 255};
    Color bgColor = {30, 30, 40, 220};
    *source_menu_ptr = menu_create(530, 30, 250, 280, 1, "Source Controls", textColor, bgColor);
    
    MenuRow *freq_row = menurow_create();
    menurow_add_interaction(freq_row,
        variableinteraction_create(&source->frequency, "Freq (Hz)     ", 0.5, 50.0, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(*source_menu_ptr, freq_row);
    
    MenuRow *amp_row = menurow_create();
    menurow_add_interaction(amp_row,
        variableinteraction_create(&source->amplitude, "Amplitude", 0.01, 0.5, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(*source_menu_ptr, amp_row);
    
    MenuRow *phase_row = menurow_create();
    menurow_add_interaction(phase_row,
        variableinteraction_create(&source->phase, "Phase (rad)", 0.0, 6.28, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(*source_menu_ptr, phase_row);
    
    MenuRow *radius_row = menurow_create();
    menurow_add_interaction(radius_row,
        variableinteraction_create(&source->radius, "Radius", 0.01, 0.2, VAR_SLIDER,
                                  NULL, NULL));
    menu_add_row(*source_menu_ptr, radius_row);
    
    MenuRow *delete_row = menurow_create();
    menurow_add_interaction(delete_row,
        variableinteraction_create(&g_dummy_delete_source, "Delete Source", 0, 1, VAR_BOOL,
                                  on_delete_source_clicked, NULL));
    menu_add_row(*source_menu_ptr, delete_row);
    
    MenuRow *deselect_row = menurow_create();
    menurow_add_interaction(deselect_row,
        variableinteraction_create(&g_dummy_deselect_source, "Deselect", 0, 1, VAR_BOOL,
                                  on_deselect_source_clicked, NULL));
    menu_add_row(*source_menu_ptr, deselect_row);
}

void free_menus(Menus *menus) {
    if (!menus) return;
    menu_free(menus->base_menu);
    menu_free(menus->mouse_menu);
    menu_free(menus->sim_menu);
    menu_free(menus->source_menu);
    free(menus);
}
