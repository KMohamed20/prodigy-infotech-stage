/*
 * Data Access Object (DAO) pour Room
 */

package com.prodigy.todoapp;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.Query;
import java.util.List;

@Dao
public interface TodoDao {
    @Insert
    void insert(Todo todo);

    @Delete
    void delete(Todo todo);

    @Query("SELECT * FROM todos")
    List<Todo> getAll();

    @Query("UPDATE todos SET completed = :completed WHERE id = :id")
    void updateCompleted(int id, boolean completed);
}
