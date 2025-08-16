/*
 * AD Task 01 - Advanced To-Do App
 * Prodigy InfoTech Internship | Khalid Ag Mohamed Aly
 * August 2025
 *
 * MainActivity: Point d'entrée de l'application Android
 */

package com.prodigy.todoapp;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.room.Room;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

import java.util.List;

public class MainActivity extends AppCompatActivity {

    private EditText taskTitleInput;
    private Spinner prioritySpinner, categorySpinner;
    private Button addButton;
    private RecyclerView tasksRecyclerView;
    private TodoAdapter todoAdapter;

    private AppDatabase db;
    private List<Todo> taskList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize views
        taskTitleInput = findViewById(R.id.taskTitleInput);
        prioritySpinner = findViewById(R.id.prioritySpinner);
        categorySpinner = findViewById(R.id.categorySpinner);
        addButton = findViewById(R.id.addButton);
        tasksRecyclerView = findViewById(R.id.tasksRecyclerView);

        // Setup RecyclerView
        tasksRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        todoAdapter = new TodoAdapter(this);
        tasksRecyclerView.setAdapter(todoAdapter);

        // Initialize Room database
        db = Room.databaseBuilder(getApplicationContext(), AppDatabase.class, "todo-db")
                .allowMainThreadQueries() // For simplicity (avoid in production)
                .build();

        // Load tasks
        loadTasks();

        // Add task button click
        addButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                addTask();
            }
        });
    }

    private void addTask() {
        String title = taskTitleInput.getText().toString().trim();
        String priority = prioritySpinner.getSelectedItem().toString();
        String category = categorySpinner.getSelectedItem().toString();

        if (title.isEmpty()) {
            Toast.makeText(this, "Entrez un titre pour la tâche", Toast.LENGTH_SHORT).show();
            return;
        }

        Todo newTask = new Todo();
        newTask.setTitle(title);
        newTask.setPriority(priority);
        newTask.setCategory(category);
        newTask.setCompleted(false);

        db.todoDao().insert(newTask);
        taskTitleInput.setText("");
        loadTasks();
        Toast.makeText(this, "Tâche ajoutée !", Toast.LENGTH_SHORT).show();
    }

    private void loadTasks() {
        taskList = db.todoDao().getAll();
        todoAdapter.setTasks(taskList);
    }
}
